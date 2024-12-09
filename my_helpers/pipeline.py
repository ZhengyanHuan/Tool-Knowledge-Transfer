import logging
import time

import torch

import configs
import model
from my_helpers.data_helpers import select_context_for_experiment, fill_missing_params, filter_keys_by_func, \
    get_default_param_dict, check_overlaps
from my_helpers.viz_helpers import viz_data, viz_test_objects_embedding
from transfer_class import Tool_Knowledge_transfer_class


def run_pipeline(loss_func=configs.loss_func, data_name=configs.data_name,
                 orig_context: dict = None, pipe_settings: dict = None, hyparams: dict = None) -> dict:
    """
    Run the transfer learning pipeline: train encoder on target tool (with shared/old obj) and source tool(with all obj)
        --> freeze encoder -->  train classifier on source tool (new obj) --> test classifier on target tool (new obj)

    :param loss_func: triplet loss as "TL" or sincere loss as "sincere"
    :param data_name: name of the dataset
    :param orig_context: context for the transfer task, such as behaviors, source tool and target tool, etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :param pipe_settings: setting for the transfer task, such as encoder_exp_name, retrain_clf etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :param hyparams: hyperparameters for the transfer task, such as encoder_output_dim, sincere_temp etc.
                         if not fully specified, will be automatically filled up by values from configs.py file.
    :return: dictionary, at least with "test_acc": accuracy from the classifier tested on target tool

    A structured way to manage and pass parameters to functions:
    1. Filters relevant keys from a dictionary based on a function's expected parameters, preventing unexpected arguments.
    2. Updates the filtered dictionary with explicitly specified values that are required for the function.
    3. Passes the dictionary to the function, allowing:
        - Default values in the target function to remain unchanged unless explicitly updated.
        - easier to identify the task specific changes without overwriting or duplicating unchanged parameters.
        - Simplifying tracking of changes to defaults arguments
            e.g., when orig_context['trail_list']==[0,1,2,3,4], full pipeline's trail_list will be [0,1,2,3,4]
                    without explicitly updating to any function with trail_list parameter
        - Avoidance of explicitly specifying all parameters, which can make the code verbose and harder to maintain.
    """
    start_time = time.time()
    pipeline_results = {}
    # %% 0. set up
    myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func, data_name=data_name)
    all_params = get_default_param_dict()
    if orig_context is not None:
        all_params.update(orig_context)
    check_overlaps(source_tool_list=all_params['source_tool_list'], target_tool_list=all_params['target_tool_list'],
                   assist_tool_list=all_params['assist_tool_list'], new_object_list=all_params['new_object_list'],
                   old_object_list=all_params['old_object_list'])
    if pipe_settings is not None:
        all_params.update(pipe_settings)
    if hyparams is not None:  # functions take hyparams as a whole dictionary,
        # so there's no need to specify one parameter for each hyperparameter
        all_params.update(hyparams)  # change default values first
        all_params.update({"hyparams": fill_missing_params(hyparams, param_model="both")})  # fill up the param list

    all_settings = filter_keys_by_func(all_params, select_context_for_experiment)
    context_dict = select_context_for_experiment(**all_settings)
    logging.debug(f"context_dict: {context_dict}")

    input_dim = 0
    data_dim = 0
    for modality in all_params['modality_list']:
        trial_batch = myclass.data_dict[all_params['behavior_list'][0]][
            all_params['target_tool_list'][0]][modality]
        x_sample = trial_batch[context_dict['clf_new_objs'][0]]['X'][0]
        input_dim += len(x_sample)
        data_dim = x_sample.shape[-1]

    if all_params['viz_dataset']:
        logging.info("👀visualize initial data ...")
        viz_params = filter_keys_by_func(all_params, viz_data)
        viz_params.update({
            "trans_cls": myclass, "encoder": None, "data_dim": data_dim,
            "assist_tool_list": context_dict['actual_assist_tools'],
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['actual_source_tools'],
            "target_tool_list": context_dict['actual_target_tools']
        })
        viz_data(**viz_params)

    # %% 1. encoder
    if all_params['retrain_encoder']:
        logging.info(f"👉 ------------ Training representation encoder using {loss_func} loss ------------ ")
        encoder_time = time.time()
        enc_params = filter_keys_by_func(all_params, myclass.train_encoder)
        enc_params.update({
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['enc_source_tools'], "target_tool_list": context_dict['enc_target_tools'],
            "trail_list": context_dict['enc_train_trail_list']
        })
        encoder = myclass.train_encoder(**enc_params)
        if all_params['save_temp_model']:
            torch.save(encoder.state_dict(), all_params['enc_pt_folder'] + all_params['encoder_pt_name'])

        logging.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                     f"min {(time.time() - encoder_time) % 60:.1f} sec.")
    else:  # look for old checkpoint
        encoder = model.encoder(input_size=input_dim, hidden_size=all_params['hyparams']['encoder_hidden_dim'],
                                output_size=all_params['hyparams']['encoder_output_dim']).to(configs.device)
        encoder.load_state_dict(torch.load(all_params['enc_pt_folder'] + all_params['encoder_pt_name'],
                                           map_location=torch.device(configs.device)))
    if all_params['viz_share_space']:
        logging.info("👀visualize embeddings in shared latent space...")
        viz_params = filter_keys_by_func(all_params, viz_data)
        viz_params.update({
            "trans_cls": myclass, "encoder": encoder, "loss_func": loss_func,
            "assist_tool_list": context_dict['enc_assist_tools'],
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['actual_source_tools'],
            "target_tool_list": context_dict['actual_target_tools']
        })
        viz_data(**viz_params)

    # %% 2. classifier
    if all_params['retrain_clf']:
        logging.info(f"👉 ------------ Training classification head ------------ ")
        clf_time = time.time()

        clf_params = filter_keys_by_func(all_params, myclass.train_classifier)
        clf_params['hyparams']['trial_val_portion'] = 0 if context_dict['clf_assist_tools'] \
            else all_params['trial_val_portion']  # allow overfit on assist data
        clf_params.update({
            'Encoder': encoder,
            "new_object_list": context_dict['clf_new_objs'],
            "source_tool_list": context_dict['clf_source_tools'],
            "trail_list": context_dict['enc_train_trail_list']
        })
        clf = myclass.train_classifier(**clf_params)
        if all_params['save_temp_model']:
            torch.save(clf.state_dict(), all_params['clf_pt_folder'] + all_params['clf_pt_name'])

        logging.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                     f"min {(time.time() - clf_time) % 60:.1f} sec.")

    else:  # look for old checkpoint
        clf = model.classifier(input_size=all_params['hyparams']['encoder_output_dim'],
                               output_size=len(context_dict['clf_new_objs'])).to(configs.device)
        clf.load_state_dict(
            torch.load(all_params['clf_pt_folder'] + all_params['clf_pt_name'],
                       map_location=torch.device(configs.device)))

    # %% 3. evaluation
    logging.info(f"👉 ------------ Evaluating the classifier ------------ ")
    eval_params = filter_keys_by_func(all_params, myclass.eval_classifier)
    eval_params.update({
        "Encoder": encoder, "Classifier": clf, "tool_list": context_dict['clf_target_tools'], "return_pred": True,
        "trail_list": context_dict['clf_val_trial_list'], "new_object_list": context_dict['clf_new_objs']
    })
    test_acc, _, pred_label_target = myclass.eval_classifier(**eval_params)
    logging.info(f"test accuracy: {test_acc * 100:.2f}%")
    logging.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

    if all_params['viz_decision_boundary']:
        logging.info("👀visualize decision boundary in shared latent space...")
        viz_params = filter_keys_by_func(all_params, viz_test_objects_embedding)
        viz_params.update({
            "transfer_class": myclass, "Encoder": encoder, "Classifier": clf, "test_accuracy": test_acc,
            "encoder_output_dim": all_params['encoder_output_dim'], "assist_tool_list": context_dict['clf_assist_tools'],
            "target_tool_list": context_dict['clf_target_tools'], "new_object_list": context_dict['clf_new_objs'],
            "source_tool_list": list(set(context_dict['clf_source_tools']) - set(context_dict['clf_assist_tools'])),
            "task_descpt": f"source:{context_dict['actual_source_tools'][0]}, "
                           f"target:{context_dict['actual_target_tools'][0]} "
                           f"encoder exp: {all_params['encoder_exp_name']}, clf exp: {all_params['clf_exp_name']}"
        })
        viz_test_objects_embedding(**viz_params)

    pipeline_results["test_acc"] = test_acc
    return pipeline_results
