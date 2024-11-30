import os
import random
import sys
import time
import logging

import numpy as np
import torch

import configs
import model
from data.helpers import viz_input_data, viz_embeddings
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
logging.basicConfig(level=logging.DEBUG, filename=log_file_path+"/log_file_main.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')

main_logger.debug(f"========================= New Run =========================")  # new log starts here
# for reproducibility
seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU.

# %% 1. task parameters
main_logger.info(f"👉 ------------ Setting up task parameters ------------ ")
behavior_list = ['3-stirring-fast']
source_tool_list = ['plastic-spoon']
# source_tool_list = ['plastic-spoon', 'wooden-fork', 'metal-whisk']
target_tool_list = ['metal-scissor']
modality_list = ['audio']
trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

old_object_list = ['cane-sugar', 'chia-seed', 'empty', 'glass-bead', 'kidney-bean', 'salt', 'split-green-pea',
                   'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['chickpea', 'detergent', 'metal-nut-bolt', 'plastic-bead', 'wheat']
# old_object_list = ['empty', 'water', 'detergent', 'chia-seed', 'cane-sugar', 'salt',
#                    'styrofoam-bead', 'split-green-pea', 'wheat', 'chickpea']
# new_object_list = ['kidney-bean', 'wooden-button', 'glass-bead', 'plastic-bead', 'metal-nut-bolt']

loss_func = "sincere"  # "TL" for triplet loss or "sincere"
data_name = 'audio_16kHz_token_down16_beh3.bin'  # downsized and flattened token vectors from behavior 3, len=744
# data_name =  "dataset_discretized.bin"
myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func, data_name=data_name)

input_dim = 0
for modality in modality_list:
    input_dim += len(myclass.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0])

encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_name = f"myclassifier_{loss_func}.pt"
retrain_encoder = False
retrain_clr = True

main_logger.info(f"input data name: {data_name}")
main_logger.info(f"behavior_list: {behavior_list}, modality_list: {target_tool_list}, trail_list: {trail_list}")
main_logger.info(f"source_tool_list: {source_tool_list}")
main_logger.info(f"target_tool_list: {target_tool_list}")
main_logger.info(f"old_object_list: {old_object_list}")
main_logger.info(f"new_object_list: {new_object_list}")
main_logger.info(f"loss_func: {loss_func}")

main_logger.info("👀visualize initial data ...")
for options in [[False, False], [True, False], [False, True]]:
    shared_only, test_only = options
    viz_input_data(shared_only=shared_only, test_only=test_only, data=myclass.data_dict, loss_func_name=loss_func,
                   behavior_list=behavior_list, source_tool_list=source_tool_list, target_tool_list=target_tool_list,
                   old_object_list=old_object_list, new_object_list=new_object_list)

start_time = time.time()
# %% 2. encoder
if retrain_encoder:
    main_logger.info(f"👉 ------------ Training representation encoder using {loss_func} loss ------------ ")
    encoder_time = time.time()
    myencoder = myclass.train_encoder(
        behavior_list=behavior_list, source_tool_list=source_tool_list, target_tool_list=target_tool_list,
        old_object_list=old_object_list, new_object_list=new_object_list, modality_list=modality_list,
        trail_list=trail_list)
    torch.save(myencoder.state_dict(), './saved_model/encoder/' + encoder_pt_name)
    main_logger.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                 f"min {(time.time() - encoder_time) % 60:.1f} sec.")

main_logger.info("👀visualize embeddings in shared latent space...")
viz_embeddings(viz_objects=["all", "shared", "test"], loss_func=loss_func, input_dim=input_dim,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               modality_list=modality_list, trail_list=trail_list, behavior_list=behavior_list,
               old_object_list=old_object_list, new_object_list=new_object_list, transfer_class=myclass)

# %% 3. classifier
if retrain_clr:
    main_logger.info(f"👉 ------------ Training classification head ------------ ")
    clf_time = time.time()

    Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                            hidden_size=configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(torch.load(
        './saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(
        behavior_list=behavior_list, source_tool_list=source_tool_list, new_object_list=new_object_list,
        modality_list=modality_list, trail_list=trail_list, Encoder=Encoder)
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

    main_logger.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                 f"min {(time.time() - clf_time) % 60:.1f} sec.")

# %% 4. evaluation
main_logger.info(f"👉 ------------ Evaluating the classifier ------------ ")
Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                        hidden_size=configs.encoder_hidden_dim).to(configs.device)
Encoder.load_state_dict(
    torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))

Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)
Classifier.load_state_dict(
    torch.load('./saved_model/classifier/' + clf_pt_name, map_location=torch.device(configs.device)))

accuracy = myclass.eval(Encoder, Classifier, behavior_list, target_tool_list, new_object_list, modality_list, trail_list)
main_logger.info(f"test accuracy: {accuracy*100:.2f}%")
main_logger.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

#%% Parameters tuning
# import random
# import train
# import numpy as np
# import torch
# import model
# import configs
# from transfer_class import Tool_Knowledge_transfer_class
# import time
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
# seed = 48
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
#
# # start_time = time.time()
#
# behavior_list = ['3-stirring-fast']
# source_tool_list = ['plastic-spoon'] #'wooden-fork', 'metal-whisk'
# target_tool_list = ['metal-scissor']
# modality_list = ['audio']
# trail_list = [0,1,2,3,4,5,6,7,8,9]
#
# train_val_list = ['detergent', 'kidney-bean', 'plastic-bead', 'chia-seed', 'salt', 'empty', 'metal-nut-bolt', 'wooden-button', 'styrofoam-bead', 'water', 'glass-bead', 'wheat']
# test_list = ['cane-sugar', 'split-green-pea', 'chickpea']
# loss_func = "TL"   # "TL" for triplet loss or "sincere"
# myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func)
#
# input_dim = 0
# for modality in modality_list:
#     input_dim+=len(myclass.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0])
#
#
# #%%
# number_of_folds = 4
# alpha_list = [0.5,1]
# lr_en_list = [0.01,0.1]
#
# best_alpha, best_lr_en = train.train_TL_k_fold(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, number_of_folds, alpha_list, lr_en_list)
# test_acc = train.train_TL_fixed_para(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, best_alpha, best_lr_en)
