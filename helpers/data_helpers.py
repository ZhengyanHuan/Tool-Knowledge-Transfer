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
    """
    :return:
    tool_order_list = ['Source Tool(All)', 'Assist Tool(Train)', 'Assist Tool(Test)',
                       'Target Tool(Train)', 'Target Tool(Test)']
    """
    return [0] * source_label_len + [1] * assist_label_train_len + [2] * assist_label_test_len + [
        3] * target_label_train_len + [4] * target_label_test_len


def restart_label_index_from_zero(labels):
    unique_sorted_values = np.sort(np.unique(labels))
    # Create a mapping from value to its sorted index
    value_to_index = {value: idx for idx, value in enumerate(unique_sorted_values)}
    return np.array([value_to_index[value] for value in labels])


def get_all_embeddings_or_data(
        trans_cls, encoder: model.encoder = None, data_dim=None,
        source_tool_list=configs.source_tool_list, assist_tool_list=configs.assist_tool_list,
        target_tool_list=configs.target_tool_list, old_object_list=configs.old_object_list,
        new_object_list=configs.new_object_list) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
    """
    :return:
        data order: source_old, source_new, assist_old, assist_new, target_old, target_new
        labels are indexed in the order of old_object_list + new_object_list
    """
    logging.debug(f"➡️ get_all_embeddings...")
    assert (encoder or data_dim) is not None
    all_emb = []
    all_labels = []
    meta_data = {}
    for t_idx, tool_list in enumerate([source_tool_list, assist_tool_list, target_tool_list]):
        for o_idx, object_list in enumerate([old_object_list, new_object_list]):
            meta_data[t_idx * o_idx + o_idx] = tool_list + object_list
            data, labels = trans_cls.get_data(tool_list=tool_list, object_list=object_list, get_labels=True)
            if data is not None:
                if encoder is not None:
                    encoded_data = encoder(data).reshape(-1, configs.encoder_output_dim).cpu().detach().numpy()
                else:
                    encoded_data = data.cpu().detach().numpy().reshape(-1, data_dim)
                labels = make_new_labels_to_curr_obj(original_labels=labels,
                                                     object_list=old_object_list + new_object_list)

                labels = labels.reshape(-1, 1)
            else:
                if encoder is not None:
                    encoded_data = np.empty((0, configs.encoder_output_dim), dtype=np.float32)
                else:
                    encoded_data = np.empty((0, data_dim), dtype=np.float32)
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
