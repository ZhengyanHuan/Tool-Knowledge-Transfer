import logging
import os
import random
import sys
import time

import numpy as np
import torch

import configs
import model
from my_helpers.data_helpers import select_context_for_experiment
from my_helpers.viz_helpers import viz_test_objects_embedding, viz_data
from transfer_class import Tool_Knowledge_transfer_class

# %%  0. setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppressing DEBUG messages from matplotlib
logging.getLogger("numexpr").setLevel(logging.WARNING)
main_logger = logging.getLogger("main_logger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
main_logger.addHandler(console_handler)  # main_logger's message will be printed on the console

log_file_path = './logs'
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
logging.basicConfig(level=logging.DEBUG, filename=log_file_path + "/log_file_main.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')

main_logger.info(f"input data name: {configs.data_name}")
main_logger.info(f"loss_func: {configs.loss_func}")

# for reproducibility
configs.set_torch_seed()

# %% 1. task setup
main_logger.debug(f"========================= New Run =========================")  # new log starts here
myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=configs.loss_func, data_name=configs.data_name)

context_dict = select_context_for_experiment()
logging.debug(f"context_dict: {context_dict}")

input_dim = 0
data_dim = 0
for modality in configs.modality_list:
    trial_batch = myclass.data_dict[configs.behavior_list[0]][configs.target_tool_list[0]][modality]
    x_sample = trial_batch[context_dict['clf_new_objs'][0]]['X'][0]
    input_dim += len(x_sample)
    data_dim = x_sample.shape[-1]

if configs.viz_dataset:
    main_logger.info("👀visualize initial data ...")
    viz_data(trans_cls=myclass, encoder=None, data_dim=data_dim,
             viz_l2_norm=configs.viz_share_space_l2_norm, assist_tool_list=context_dict['actual_assist_tools'],
             new_object_list=context_dict['enc_new_objs'], old_object_list=context_dict['enc_old_objs'],
             source_tool_list=context_dict['actual_source_tools'], target_tool_list=context_dict['actual_target_tools'])

start_time = time.time()
# %% 2. encoder
if configs.retrain_encoder:
    main_logger.info(f"👉 ------------ Training representation encoder using {configs.loss_func} loss ------------ ")
    encoder_time = time.time()
    myencoder = myclass.train_encoder(
        source_tool_list=context_dict['enc_source_tools'], target_tool_list=context_dict['enc_target_tools'],
        new_object_list=context_dict['enc_new_objs'], old_object_list=context_dict['enc_old_objs'],
        trail_list=context_dict['enc_train_trail_list'])
    torch.save(myencoder.state_dict(), './saved_model/encoder/' + configs.encoder_pt_name)
    main_logger.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                     f"min {(time.time() - encoder_time) % 60:.1f} sec.")

if configs.viz_share_space:
    Encoder = model.encoder(input_size=input_dim).to(configs.device)
    Encoder.load_state_dict(torch.load('./saved_model/encoder/' + configs.encoder_pt_name,
                                       map_location=torch.device(configs.device)))
    main_logger.info("👀visualize embeddings in shared latent space...")
    viz_data(trans_cls=myclass, encoder=Encoder, loss_func=configs.loss_func,
             viz_l2_norm=configs.viz_share_space_l2_norm, assist_tool_list=context_dict['enc_assist_tools'],
             new_object_list=context_dict['enc_new_objs'], old_object_list=context_dict['enc_old_objs'],
             source_tool_list=context_dict['actual_source_tools'], target_tool_list=context_dict['actual_target_tools'])
# %% 3. classifier
if configs.retrain_clr:
    main_logger.info(f"👉 ------------ Training classification head ------------ ")
    clf_time = time.time()

    Encoder = model.encoder(input_size=input_dim).to(configs.device)
    Encoder.load_state_dict(torch.load('./saved_model/encoder/' + configs.encoder_pt_name,
                                       map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(Encoder=Encoder, trial_val_portion=configs.trial_val_portion,
                                            source_tool_list=context_dict['clf_source_tools'],
                                            new_object_list=context_dict['clf_new_objs'],
                                            trail_list=context_dict['enc_train_trail_list'])
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + configs.clf_pt_name)

    main_logger.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                     f"min {(time.time() - clf_time) % 60:.1f} sec.")

# %% 4. evaluation
main_logger.info(f"👉 ------------ Evaluating the classifier ------------ ")
Encoder = model.encoder(input_size=input_dim).to(configs.device)
Encoder.load_state_dict(
    torch.load('./saved_model/encoder/' + configs.encoder_pt_name, map_location=torch.device(configs.device)))

Classifier = model.classifier(output_size=len(context_dict['clf_new_objs'])).to(configs.device)
Classifier.load_state_dict(
    torch.load('./saved_model/classifier/' + configs.clf_pt_name, map_location=torch.device(configs.device)))

accuracy, _, pred_label_target = myclass.eval(Encoder=Encoder, Classifier=Classifier,  # evaluate target tool
                                              tool_list=context_dict['clf_target_tools'], return_pred=True,
                                              new_object_list=context_dict['clf_new_objs'])
main_logger.info(f"test accuracy: {accuracy * 100:.2f}%")
main_logger.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

main_logger.info("👀visualize decision boundary in shared latent space...")
if len(context_dict['actual_source_tools']) == 1:
    source_tools_descpt = context_dict['actual_source_tools'][0]
else:
    source_tools_descpt = f"{len(context_dict['actual_source_tools'])} tools"
viz_test_objects_embedding(
    transfer_class=myclass, Encoder=Encoder, Classifier=Classifier, test_accuracy=accuracy,
    pred_label_target=pred_label_target, encoder_output_dim=configs.encoder_output_dim,
    assist_tool_list=context_dict['clf_assist_tools'], new_object_list=context_dict['clf_new_objs'],
    source_tool_list=list(set(context_dict['clf_source_tools']) - set(context_dict['clf_assist_tools'])),
    target_tool_list=context_dict['clf_target_tools'],
    viz_l2_norm=configs.viz_share_space_l2_norm,
    task_descpt=f"source:{source_tools_descpt}, target:{context_dict['actual_target_tools'][0]} "
                f"encoder exp: {configs.encoder_exp_name}, clf exp: {configs.clf_exp_name}")
