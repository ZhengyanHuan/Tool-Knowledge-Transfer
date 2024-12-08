import copy
import inspect
import logging
from typing import Tuple, List, Union, Dict

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


def train_test_split_by_trials(source_data, truth_source, target_data=None, truth_target=None,
                               trial_list=configs.trail_list, trial_val_portion=0.2,
                               random_trials=configs.randomize_trials):
    """target data should only have old objects, object dimension is order by old_obj_list to new_obj_list"""
    logging.debug(f"➡️train_test_split_by_trials... trial_list: {trial_list},"
                  f"trial_val_portion: {trial_val_portion}, random_trials: {random_trials}")
    if trial_val_portion == 0:
        return {
            "source_data_train": source_data,
            "source_data_val": None,
            "truth_source_train": truth_source,
            "truth_source_val": None,
            "target_data_train": target_data,
            "target_data_val": None,
            "truth_target_train": truth_target,
            "truth_target_val": None,

        }
    device = source_data.device
    num_trials = source_data.shape[3]
    assert num_trials == len(trial_list), "Mismatch between number of trials and trial_list length."
    num_val_trials = int(num_trials * trial_val_portion)
    num_tr_trials = num_trials - num_val_trials

    source_data, truth_source = source_data.detach().cpu().numpy(), truth_source.detach().cpu().numpy()
    if target_data is not None:
        target_data, truth_target = target_data.detach().cpu().numpy(), truth_target.detach().cpu().numpy()

    if not random_trials:
        source_data_train = source_data[:, :, :, :num_tr_trials]
        source_data_val = source_data[:, :, :, -num_val_trials:]
        truth_source_train = truth_source[:, :, :, :num_tr_trials]
        truth_source_val = truth_source[:, :, :, -num_val_trials:]
        if target_data is not None:
            target_data_train = target_data[:, :, :, :num_tr_trials]
            target_data_val = target_data[:, :, :, -num_val_trials:]
            truth_target_train = truth_target[:, :, :, :num_tr_trials]
            truth_target_val = truth_target[:, :, :, -num_val_trials:]


    else:
        configs.set_torch_seed()
        # TODO: make tirla indes, and sample from that
        # Initialize data structures for train and validation sets
        source_data_train = np.empty_like(source_data[:, :, :, :num_tr_trials])
        source_data_val = np.empty_like(source_data[:, :, :, -num_val_trials:])
        truth_source_train = np.empty_like(truth_source[:, :, :, :num_tr_trials])
        truth_source_val = np.empty_like(truth_source[:, :, :, -num_val_trials:])
        if target_data is not None:
            num_old_obj = target_data.shape[2]
            target_data_train = np.empty_like(target_data[:, :, :, :num_tr_trials])
            target_data_val = np.empty_like(target_data[:, :, :, -num_val_trials:])
            truth_target_train = np.empty_like(truth_target[:, :, :, :num_tr_trials])
            truth_target_val = np.empty_like(truth_target[:, :, :, -num_val_trials:])
        else:
            num_old_obj = 0
            target_data_train, target_data_val, truth_target_val, truth_target_train = None, None, None, None

        num_all_obj = source_data.shape[2]
        for obj_idx in range(num_all_obj):
            # Randomly select trials for this object
            val_trials = np.random.choice(trial_list, size=num_val_trials, replace=False)
            train_trials = np.setdiff1d(trial_list, val_trials)
            # Select the sampled trials for each object
            source_data_train[:, :, obj_idx] = source_data[:, :, obj_idx, train_trials]
            source_data_val[:, :, obj_idx] = source_data[:, :, obj_idx, val_trials]
            truth_source_train[:, :, obj_idx] = truth_source[:, :, obj_idx, train_trials]
            truth_source_val[:, :, obj_idx] = truth_source[:, :, obj_idx, val_trials]
            if target_data is not None:  # old obj only
                if obj_idx < num_old_obj:
                    target_data_train[:, :, obj_idx] = target_data[:, :, obj_idx, train_trials]
                    target_data_val[:, :, obj_idx] = target_data[:, :, obj_idx, val_trials]
                    truth_target_train[:, :, obj_idx] = truth_target[:, :, obj_idx, train_trials]
                    truth_target_val[:, :, obj_idx] = truth_target[:, :, obj_idx, val_trials]

    logging.debug(f"train test split shapes: source_data_train: {source_data_train.shape},"
                  f"truth_source_train: {truth_source_train.shape}, source_data_val: {source_data_val.shape},"
                  f"truth_source_val: {truth_source_val.shape}")
    if target_data is not None:
        logging.debug(f"target_data_train: {target_data_train.shape}, truth_target_train: {truth_target_train.shape}, "
                      f"target_data_val: {target_data_val.shape}, truth_target_val: {truth_target_val.shape}")
    data_dict = {
        "source_data_train": torch.tensor(source_data_train).to(device),
        "source_data_val": torch.tensor(source_data_val).to(device),
        "truth_source_train": torch.tensor(truth_source_train).to(device),
        "truth_source_val": torch.tensor(truth_source_val).to(device)}
    if target_data is not None:
        data_dict.update({
            "target_data_train": torch.tensor(target_data_train).to(device),
            "target_data_val": torch.tensor(target_data_val).to(device),
            "truth_target_train": torch.tensor(truth_target_train).to(device),
            "truth_target_val": torch.tensor(truth_target_val).to(device)})
    else:
        data_dict.update({
            "target_data_train": None,
            "target_data_val": None,
            "truth_target_train": None,
            "truth_target_val": None})
    return data_dict


def make_new_labels_to_curr_obj(original_labels: Union[torch.Tensor, np.array], object_list: list):
    """take original label from the data set, assign new labels by all objects in old+new order"""
    if original_labels is None:
        return original_labels
    obj_flattened = [SORTED_DATA_OBJ_LIST[item] for item in original_labels.flatten()]
    relative_labels = np.array([object_list.index(obj) for obj in obj_flattened])
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
        behavior_list=configs.behavior_list, modality_list=configs.modality_list, trail_list=configs.trail_list,
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
            data, labels = trans_cls.get_data(tool_list=tool_list, object_list=object_list, get_labels=True,
                                              behavior_list=behavior_list, modality_list=modality_list,
                                              trail_list=trail_list)
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
            logging.debug(f"trial data shape for {tool_list} and {len(object_list)} objects: {encoded_data.shape}")
            logging.debug(f"trial labels shape for {tool_list} and {len(object_list)} objects: {labels.shape}")

    return all_emb, all_labels, meta_data


def select_context_for_experiment(
        encoder_exp_name=configs.encoder_exp_name, clf_exp_name=configs.clf_exp_name, exp_pred_obj=configs.exp_pred_obj,
        source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
        assist_tool_list=configs.assist_tool_list, old_object_list=configs.old_object_list,
        new_object_list=configs.new_object_list, trial_list=configs.trail_list,
        enc_trial_list=configs.enc_trial_list) -> Dict[str, List[str]]:
    assert encoder_exp_name in ["default", "all", "assist", "assist_no-target", "baseline1", "baseline2",
                                "baseline2-all"]
    assert clf_exp_name in ["default", "assist"]
    assert exp_pred_obj in ['all', 'new']
    logging.debug(f"experiment: encoder: {encoder_exp_name}, clf: {clf_exp_name}, predict objects: {exp_pred_obj}")
    all_object_list = old_object_list + new_object_list

    exp_context_dict = {
        'exp_pred_obj': exp_pred_obj,
        'actual_source_tools': source_tool_list,
        'actual_target_tools': target_tool_list,
        'actual_assist_tools': [],
        'enc_source_tools': source_tool_list,  # source tool that has all objects
        'enc_target_tools': target_tool_list,  # target tool that only has new object
        'enc_assist_tools': [],  # assist tool(s) that only has old objects
        'enc_old_objs': old_object_list,  # share objects for source and target
        'enc_new_objs': new_object_list,  # objects only for source
        'enc_train_trail_list': trial_list,
        'clf_val_trial_list': trial_list,

        'clf_source_tools': source_tool_list,  # the tool(s) used to train the clf
        'clf_target_tools': target_tool_list,  # the tool used to test the clf
        'clf_assist_tools': [],  # the assist tool used to train the clf
        'clf_old_objs': old_object_list,  # objects used to train and test the classifier
        'clf_new_objs': new_object_list,  # objects used to train and test the clf
    }
    if exp_pred_obj == "all":  # predict all objects
        exp_context_dict['clf_new_objs'] = all_object_list
        exp_context_dict['clf_old_objs'] = []

    # ---- encoder source tools
    if encoder_exp_name == "all":  # use all other tools to train encoder
        exp_context_dict['enc_source_tools'] = source_tool_list + assist_tool_list
        exp_context_dict['actual_assist_tools'] = assist_tool_list
    elif encoder_exp_name in ["assist", "assist_no-target"]:
        # besides source, use old object from assist tools to train encoder
        exp_context_dict['actual_assist_tools'] = assist_tool_list
        exp_context_dict['enc_assist_tools'] = assist_tool_list
        # balance the source tool and target tool number, in case assist and target outnumber source
        #  so much that source&source pairs are considered less during contrastive learning
        tool_gap = len(assist_tool_list) + len(target_tool_list) - len(source_tool_list)
        if tool_gap > 0:
            exp_context_dict['enc_source_tools'] = source_tool_list * (tool_gap + 1)
    elif encoder_exp_name == "baseline1":  # train on target tool only
        exp_context_dict['actual_source_tools'] = target_tool_list
        exp_context_dict['enc_source_tools'] = target_tool_list
        exp_context_dict['enc_train_trail_list'] = enc_trial_list
        exp_context_dict['clf_val_trial_list'] = list(set(configs.trail_list) - set(enc_trial_list))
    elif encoder_exp_name == "baseline2-all":  # train on all tools that are not target tool
        print(source_tool_list + assist_tool_list)
        exp_context_dict['actual_source_tools'] = source_tool_list + assist_tool_list
        exp_context_dict['enc_source_tools'] = source_tool_list + assist_tool_list

    # ---- encoder target tools and object selection
    if encoder_exp_name == "assist":
        exp_context_dict['enc_target_tools'] = target_tool_list + assist_tool_list
    if encoder_exp_name == "assist_no-target":  # true zero shot learning, no target information during rep learning
        assert len(assist_tool_list) != 0
        exp_context_dict['enc_target_tools'] = assist_tool_list
        # all objects for training encoder
        exp_context_dict['enc_new_objs'] = all_object_list
        exp_context_dict['enc_old_objs'] = []
    elif "baseline" in encoder_exp_name:  # no transfer, so there's no new object for encoder,
        exp_context_dict['enc_target_tools'] = []
        if exp_pred_obj == "new":
            exp_context_dict['enc_new_objs'] = []
            exp_context_dict['enc_old_objs'] = new_object_list
            exp_context_dict['clf_new_objs'] = new_object_list
            exp_context_dict['clf_old_objs'] = []
        elif exp_pred_obj == "all":
            exp_context_dict['enc_new_objs'] = []
            exp_context_dict['enc_old_objs'] = all_object_list
            exp_context_dict['clf_new_objs'] = all_object_list
            exp_context_dict['clf_old_objs'] = []

    # ---- classifier source
    if clf_exp_name == "assist" and "assist" in encoder_exp_name:
        # besides source, use new objects from assist tools to train clf
        exp_context_dict['clf_source_tools'] = source_tool_list + assist_tool_list
        exp_context_dict['clf_assist_tools'] = assist_tool_list
    # regardless of what clf_exp_name is
    if encoder_exp_name == "baseline1":
        exp_context_dict['clf_source_tools'] = target_tool_list
    elif encoder_exp_name == "baseline2":
        exp_context_dict['clf_source_tools'] = source_tool_list
    elif encoder_exp_name == "baseline2-all":
        exp_context_dict['clf_source_tools'] = source_tool_list + assist_tool_list
    elif encoder_exp_name == "all":
        exp_context_dict['clf_source_tools'] = source_tool_list + assist_tool_list

    # ---- classifier target
    # so far, always test on the target tool

    return exp_context_dict


def fill_missing_params(hyparams: dict, param_model="classifier"):
    """ fill up the missing parameters using values from configs """
    if hyparams is None:
        hyparams = {}
    overlap_list = ["trial_val_portion", "encoder_output_dim", 'encoder_hidden_dim', 'trial_val_portion',
                    'randomize_trials', 'tolerance', 'smooth_wind_size']
    clf_list = ["epoch_classifier", 'lr_classifier', "early_stop_patience_clf", 'clf_tolerance']
    enc_list = ['encoder_hidden_dim', "epoch_encoder", 'lr_encoder', "early_stop_patience_enc",
                "TL_margin", "sincere_temp", "pairs_per_batch_per_object"]

    if param_model == "classifier":
        param_names = overlap_list + clf_list
    elif param_model == "encoder":
        param_names = overlap_list + enc_list
    elif param_model == "both":
        param_names = overlap_list + clf_list + enc_list
    else:
        raise Exception(f"param_model {param_model} not available")

    for name in param_names:
        if name not in hyparams.keys():
            hyparams[name] = getattr(configs, name)
    return hyparams


def fill_missing_context(context_dict):
    if context_dict is None:
        context_dict = {}
    context_names = ['behavior_list', 'modality_list', 'trail_list',
                     'old_object_list', 'new_object_list', 'all_object_list',
                     'source_tool_list', 'assist_tool_list', 'target_tool_list']
    for name in context_names:
        if name not in context_dict.keys():
            context_dict[name] = getattr(configs, name)
    return context_dict


def fill_missing_pipeline_settings(pipeline_settings):
    if pipeline_settings is None:
        pipeline_settings = {}
    setting_names = [
        'encoder_exp_name', 'clf_exp_name', 'exp_pred_obj',  # experiment info
        'save_temp_model', 'enc_pt_folder', 'clf_pt_folder', 'encoder_pt_name', 'clf_pt_name',  # save models
        'retrain_encoder', 'retrain_clf',  # train models from scratch
        'viz_dataset', 'viz_share_space', 'viz_l2_norm', 'viz_decision_boundary', 'plot_learning', 'save_fig'  # viz
    ]
    for name in setting_names:
        if name not in pipeline_settings.keys():
            pipeline_settings[name] = getattr(configs, name)
    return pipeline_settings


def filter_keys_by_func(param_dict, function):
    explicit_args = inspect.signature(function).parameters.keys()
    return copy.deepcopy({k: v for k, v in param_dict.items() if k in explicit_args})
