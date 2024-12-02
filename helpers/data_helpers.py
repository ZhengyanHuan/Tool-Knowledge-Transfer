import copy
import logging
from typing import Tuple, List, Union

import numpy as np
import torch

import configs
import model

# for input data from data files, the labels were created in this order
SORTED_DATA_OBJ_LIST = sorted(['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
                               'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea', 'kidney-bean',
                               'wooden-button', 'plastic-bead', 'glass-bead', 'metal-nut-bolt'])
TOOL_GROUPS = ['source', 'assist', 'target']
OBJ_GROUPS = ['old', 'new']


def sanity_check_data_labels(data_dict: dict):
    """ check if the data's labels were created in the order of the sorted object list: SORTED_DATA_OBJ_LIST"""
    mismatch = False
    for behavior in data_dict.keys():
        for tool in data_dict[behavior].keys():
            for modality in data_dict[behavior][tool].keys():
                for obj in data_dict[behavior][tool][modality].keys():
                    trail_batch = data_dict[behavior][tool][modality][obj]
                    trail_batch_label = np.squeeze(trail_batch['Y'])
                    for trial_num, label in enumerate(trail_batch_label):
                        try:
                            assert obj == SORTED_DATA_OBJ_LIST[label]
                        except AssertionError:
                            mismatch = True
                            print(f"object does not match label: {behavior}_{tool}_{modality}_{obj}-trial-{trial_num}: "
                                  f"{label} instead of {SORTED_DATA_OBJ_LIST.index(obj)}")
    if mismatch:
        raise AssertionError


def stack_input_data(X_array, Y_array, behavior_list, tool_list, obj_list, data):
    """stack new data to X_array, Y_array"""
    X_array = [X_array] if len(X_array) != 0 else []
    Y_array = [Y_array] if len(Y_array) != 0 else []
    meta_data = {b: {t: {} for t in tool_list} for b in behavior_list}
    for b_idx, b in enumerate(behavior_list):
        for t_idx, t in enumerate(tool_list):
            for o_idx, o in enumerate(obj_list):
                X_array.append(data[b][t]['audio'][o]['X'])
                Y_array.append(data[b][t]['audio'][o]['Y'])
                meta_data[b][t][o] = len(data[b][t]['audio'][o]['Y'])
    logging.debug(f"data input meta_data: {meta_data}")
    return np.vstack(X_array), np.vstack(Y_array)


def make_new_labels_to_curr_obj(original_labels: Union[torch.Tensor, np.array], object_list: list):
    """take original label from the data set, assign new labels by all objects in old+new order"""
    if original_labels is None:
        return original_labels
    obj_flattened = [SORTED_DATA_OBJ_LIST[item] for item in original_labels.flatten()]
    relative_labels = np.array([object_list.index(obj) for obj in obj_flattened])
    logging.debug(f"relative_labels: \n    {relative_labels}")
    return relative_labels.reshape(original_labels.shape)


def create_tool_idx_list(source_label_len=0, assist_label_train_len=0,
                         assist_label_test_len=0, target_label_train_len=0, target_label_test_len=0) -> list:
    # tool_order_list = ['Source Tool(All)', 'Assist Tool(Train)', 'Assist Tool(Test)',
    #                    'Target Tool(Train)', 'Target Tool(Test)']
    return [0] * source_label_len + [1] * assist_label_train_len + [2] * assist_label_test_len + [3] * target_label_train_len + [4] * target_label_test_len


def restart_label_index_from_zero(labels):
    unique_sorted_values = np.sort(np.unique(labels))
    # Create a mapping from value to its sorted index
    value_to_index = {value: idx for idx, value in enumerate(unique_sorted_values)}
    return np.array([value_to_index[value] for value in labels])


def get_all_embeddings(
        trans_cls, encoder: model.encoder,
        source_tool_list=configs.source_tool_list, assist_tool_list=configs.assist_tool_list,
        target_tool_list=configs.target_tool_list, old_object_list=configs.old_object_list,
        new_object_list=configs.new_object_list) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """

    :param trans_cls:
    :param encoder:
    :param source_tool_list:
    :param assist_tool_list:
    :param target_tool_list:
    :param old_object_list:
    :param new_object_list:
    :return:
        data order: source_old, source_new, assist_old, assist_new, target_old, target_new
        labels are indexed in the order of old_object_list + new_object_list
    """
    logging.debug(f"➡️ get_all_embeddings...")
    all_emb = []
    all_labels = []
    meta_data = {}
    for t_idx, tool_list in enumerate([source_tool_list, assist_tool_list, target_tool_list]):
        for o_idx, object_list in enumerate([old_object_list, new_object_list]):
            meta_data[t_idx * o_idx + o_idx] = tool_list + object_list
            data, labels = trans_cls.get_data(tool_list=tool_list, object_list=object_list, get_labels=True)
            if data is not None:
                encoded_data = encoder(data).reshape(-1, configs.encoder_output_dim).cpu().detach().numpy()
                labels = make_new_labels_to_curr_obj(original_labels=labels,
                                                     object_list=old_object_list + new_object_list)
                labels = labels.reshape(-1, 1)
            else:
                encoded_data = np.empty((0, configs.encoder_output_dim), dtype=np.float32)
                labels = np.empty((0, 1), dtype=np.int64)
            all_emb.append(encoded_data)
            all_labels.append(labels)
            logging.debug(f"encoded_data shape for {tool_list} and {len(object_list)} objects: {encoded_data.shape}")
            logging.debug(f"labels shape for {tool_list} and {len(object_list)} objects: {labels.shape}")
    logging.debug(f"groups of data: {len(all_emb)}")
    logging.debug(f"groups of labels: {len(all_labels)}")

    return all_emb, all_labels, meta_data


def select_context_for_experiment(encoder_exp_name="", clf_exp_name=""):
    source_tool_list = copy.copy(configs.source_tool_list)
    target_tool_list = copy.copy(configs.target_tool_list)
    assist_tool_list = copy.copy(configs.assist_tool_list)

    encoder_source_tool_list = source_tool_list
    encoder_target_tool_list = target_tool_list
    clf_source_tool_list = source_tool_list
    clf_target_tool_list = target_tool_list

    if encoder_exp_name == "source-all":  # use all other tools to train encoder
        encoder_source_tool_list = source_tool_list + assist_tool_list
    elif encoder_exp_name == "source-assist":  # besides source, use old object from assist tool to train encoder
        # balance the source tool and target tool number
        tool_gap = len(assist_tool_list) + len(target_tool_list) - len(source_tool_list)
        if tool_gap > 0:
            encoder_source_tool_list *= (tool_gap + 1)
        encoder_target_tool_list = target_tool_list + assist_tool_list

    if clf_exp_name == "source-assist":  # besides source, use new object from assist tool to train clf
        clf_source_tool_list = source_tool_list + assist_tool_list

    return encoder_source_tool_list, encoder_target_tool_list, clf_source_tool_list, clf_target_tool_list

# def select_embedding_by_context(all_emb: List[np.ndarray], all_labels: List[np.ndarray],
#                                 meta_data: dict, tool_groups: List[List[str]], object_groups: List[List[str]]) \
#         -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     selected_emb = []
#     selected_label = []
#     for tool_list in tool_groups:
#         for obj_list in object_groups:
#             data_idx = None
#             for idx, context in meta_data.items():
#                 if context == tool_list + obj_list:
#                     data_idx = idx
#             if data_idx is not None:
#                 selected_emb.append(all_emb[meta_data[data_idx]])
#                 selected_label.append(all_labels[data_idx])
#             else:
#                 raise Exception(f"can't find context: {tool_list}: {obj_list}")
#     return selected_emb, selected_label
