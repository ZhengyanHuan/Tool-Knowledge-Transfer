import logging
import sys
import time

import configs

import torch

import model
from my_helpers.data_helpers import select_context_for_experiment, fill_missing_params, fill_missing_context, \
    fill_missing_pipeline_settings, filter_keys_by_func
from my_helpers.viz_helpers import viz_data, viz_test_objects_embedding
from transfer_class import Tool_Knowledge_transfer_class


def run_pipeline(loss_func=configs.loss_func, data_name=configs.data_name,
                 orig_context=None, pipe_settings=None, hyparams=None):
    start_time = time.time()
    pipeline_results = {}
    # %% 0. set up
    myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func, data_name=data_name)
    # fill up necessary params using configs file
    orig_context = fill_missing_context(context_dict=orig_context)
    pipe_settings = fill_missing_pipeline_settings(pipeline_settings=pipe_settings)
    hyparams = fill_missing_params(hyparams=hyparams, param_model="both")
    context_pip_params = {**pipe_settings, **orig_context}
    model_params = {"hyparams": hyparams, **orig_context}

    all_settings = filter_keys_by_func(context_pip_params, select_context_for_experiment)
    context_dict = select_context_for_experiment(**all_settings)
    logging.debug(f"context_dict: {context_dict}")

    input_dim = 0
    data_dim = 0
    for modality in orig_context['modality_list']:
        trial_batch = myclass.data_dict[orig_context['behavior_list'][0]][
            orig_context['target_tool_list'][0]][modality]
        x_sample = trial_batch[context_dict['clf_new_objs'][0]]['X'][0]
        input_dim += len(x_sample)
        data_dim = x_sample.shape[-1]

    if pipe_settings['viz_dataset']:
        logging.info("👀visualize initial data ...")
        viz_params = filter_keys_by_func(context_pip_params, viz_data)
        viz_params.update({
            "trans_cls": myclass, "encoder": None, "data_dim": data_dim,
            "assist_tool_list": context_dict['actual_assist_tools'],
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['actual_source_tools'],
            "target_tool_list": context_dict['actual_target_tools']
        })
        viz_data(**viz_params)

    # %% 1. encoder
    if pipe_settings['retrain_encoder']:
        logging.info(f"👉 ------------ Training representation encoder using {loss_func} loss ------------ ")
        encoder_time = time.time()
        enc_params = filter_keys_by_func(model_params, myclass.train_encoder)
        enc_params.update({
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['enc_source_tools'], "target_tool_list": context_dict['enc_target_tools'],
            "trail_list": context_dict['enc_train_trail_list']
        })
        encoder = myclass.train_encoder(**enc_params)
        if pipe_settings['save_temp_model']:
            torch.save(encoder.state_dict(), pipe_settings['enc_pt_folder'] + pipe_settings['encoder_pt_name'])

        logging.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                     f"min {(time.time() - encoder_time) % 60:.1f} sec.")
    else:  # look for old checkpoint
        encoder = model.encoder(input_size=input_dim).to(configs.device)
        encoder.load_state_dict(torch.load(pipe_settings['enc_pt_folder'] + pipe_settings['encoder_pt_name'],
                                           map_location=torch.device(configs.device)))

    if pipe_settings['viz_share_space']:
        logging.info("👀visualize embeddings in shared latent space...")
        viz_params = filter_keys_by_func(context_pip_params, viz_data)
        viz_params.update({
            "trans_cls": myclass, "encoder": encoder, "loss_func": loss_func,
            "assist_tool_list": context_dict['enc_assist_tools'],
            "new_object_list": context_dict['enc_new_objs'], "old_object_list": context_dict['enc_old_objs'],
            "source_tool_list": context_dict['actual_source_tools'],
            "target_tool_list": context_dict['actual_target_tools']
        })
        viz_data(**viz_params)

    # %% 2. classifier
    if pipe_settings['retrain_clf']:
        logging.info(f"👉 ------------ Training classification head ------------ ")
        clf_time = time.time()

        clf_params = filter_keys_by_func(model_params, myclass.train_classifier)
        clf_params['hyparams']['trial_val_portion'] = 0 if context_dict['clf_assist_tools'] \
            else hyparams['trial_val_portion']  # allow overfit on assist data
        clf_params.update({
            'Encoder': encoder,
            "new_object_list": context_dict['clf_new_objs'],
            "source_tool_list": context_dict['clf_source_tools'],
            "trail_list": context_dict['enc_train_trail_list']
        })
        clf = myclass.train_classifier(**clf_params)
        if pipe_settings['save_temp_model']:
            torch.save(clf.state_dict(), pipe_settings['clf_pt_folder'] + pipe_settings['clf_pt_name'])

        logging.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                     f"min {(time.time() - clf_time) % 60:.1f} sec.")

    else:  # look for old checkpoint
        clf = model.classifier(output_size=len(context_dict['clf_new_objs'])).to(configs.device)
        clf.load_state_dict(
            torch.load(pipe_settings['clf_pt_folder'] + pipe_settings['clf_pt_name'],
                       map_location=torch.device(configs.device)))

    # %% 3. evaluation
    logging.info(f"👉 ------------ Evaluating the classifier ------------ ")
    eval_params = filter_keys_by_func(model_params, myclass.eval_classifier)
    eval_params.update({
        "Encoder": encoder, "Classifier": clf, "tool_list": context_dict['clf_target_tools'], "return_pred": True,
        "trail_list": context_dict['clf_val_trial_list'], "new_object_list": context_dict['clf_new_objs']
    })
    test_acc, _, pred_label_target = myclass.eval_classifier(**eval_params)
    logging.info(f"test accuracy: {test_acc * 100:.2f}%")
    logging.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

    if pipe_settings['viz_decision_boundary']:
        logging.info("👀visualize decision boundary in shared latent space...")
        viz_params = filter_keys_by_func(context_pip_params, viz_test_objects_embedding)
        viz_params.update({
            "transfer_class": myclass, "Encoder": encoder, "Classifier": clf, "test_accuracy": test_acc,
            "encoder_output_dim": hyparams['encoder_output_dim'], "assist_tool_list": context_dict['clf_assist_tools'],
            "target_tool_list": context_dict['clf_target_tools'], "new_object_list": context_dict['clf_new_objs'],
            "source_tool_list": list(set(context_dict['clf_source_tools']) - set(context_dict['clf_assist_tools'])),
            "task_descpt": f"source:{context_dict['actual_source_tools'][0]}, "
                           f"target:{context_dict['actual_target_tools'][0]} "
                           f"encoder exp: {pipe_settings['encoder_exp_name']}, clf exp: {pipe_settings['clf_exp_name']}"
        })
        viz_test_objects_embedding(**viz_params)

    pipeline_results["test_acc"] = test_acc
    return pipeline_results
