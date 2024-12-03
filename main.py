import logging
import os
import random
import time

import numpy as np
import torch

import configs
import model
from helpers.data_helpers import select_context_for_experiment
from helpers.viz_helpers import viz_test_objects_embedding, viz_data
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
main_logger.info(f"behavior_list: {configs.behavior_list}, modality_list: {configs.target_tool_list}, "
                 f"trail_list: {configs.trail_list}")
main_logger.info(f"source_tool_list: {configs.source_tool_list}")
main_logger.info(f"target_tool_list: {configs.target_tool_list}")
main_logger.info(f"old_object_list: {configs.old_object_list}")
main_logger.info(f"new_object_list: {configs.new_object_list}")
main_logger.info(f"loss_func: {configs.loss_func}")

# for reproducibility
configs.set_torch_seed()

# %% 1. task setup
main_logger.debug(f"========================= New Run =========================")  # new log starts here
myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=configs.loss_func, data_name=configs.data_name)

(encoder_source_tool_list, encoder_target_tool_list,
 clf_source_tool_list, clf_target_tool_list) = select_context_for_experiment(configs.encoder_exp_name, configs.clf_exp_name)

input_dim = 0
data_dim = 0
for modality in configs.modality_list:
    obj_trial_batch = myclass.data_dict[configs.behavior_list[0]][configs.target_tool_list[0]][modality]
    x_sample = obj_trial_batch[configs.old_object_list[0]]['X'][0]
    input_dim += len(x_sample)
    data_dim = x_sample.shape[-1]

if configs.viz_dataset:
    main_logger.info("👀visualize initial data ...")
    viz_data(trans_cls=myclass, data_dim=data_dim, encoder=None, viz_l2_norm=False)

start_time = time.time()
# %% 2. encoder
if configs.retrain_encoder:
    main_logger.info(f"👉 ------------ Training representation encoder using {configs.loss_func} loss ------------ ")
    encoder_time = time.time()
    configs.set_torch_seed()
    myencoder = myclass.train_encoder(source_tool_list=encoder_source_tool_list,
                                      target_tool_list=encoder_target_tool_list)
    torch.save(myencoder.state_dict(), './saved_model/encoder/' + configs.encoder_pt_name)
    main_logger.info(f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} "
                     f"min {(time.time() - encoder_time) % 60:.1f} sec.")

if configs.viz_share_space:
    Encoder = model.encoder(input_size=input_dim).to(configs.device)
    Encoder.load_state_dict(torch.load('./saved_model/encoder/' + configs.encoder_pt_name,
                                       map_location=torch.device(configs.device)))
    main_logger.info("👀visualize embeddings in shared latent space...")
    viz_data(trans_cls=myclass, encoder=Encoder, viz_l2_norm=configs.viz_share_space_l2_norm)
# %% 3. classifier
if configs.retrain_clr:
    main_logger.info(f"👉 ------------ Training classification head ------------ ")
    clf_time = time.time()

    Encoder = model.encoder(input_size=input_dim).to(configs.device)
    Encoder.load_state_dict(torch.load('./saved_model/encoder/' + configs.encoder_pt_name,
                                       map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(Encoder=Encoder,
                                            source_tool_list=clf_source_tool_list)
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + configs.clf_pt_name)

    main_logger.info(f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} "
                     f"min {(time.time() - clf_time) % 60:.1f} sec.")

# %% 4. evaluation
main_logger.info(f"👉 ------------ Evaluating the classifier ------------ ")
Encoder = model.encoder(input_size=input_dim).to(configs.device)
Encoder.load_state_dict(
    torch.load('./saved_model/encoder/' + configs.encoder_pt_name, map_location=torch.device(configs.device)))

Classifier = model.classifier().to(configs.device)
Classifier.load_state_dict(
    torch.load('./saved_model/classifier/' + configs.clf_pt_name, map_location=torch.device(configs.device)))

accuracy, _, pred_label_target = myclass.eval(Encoder=Encoder, Classifier=Classifier,  # evaluate target tool
                                              tool_list=clf_target_tool_list, return_pred=True)
main_logger.info(f"test accuracy: {accuracy * 100:.2f}%")
main_logger.info(f"⏱️total time used: {round((time.time() - start_time) // 60)} "
                 f"min {(time.time() - start_time) % 60:.1f} sec.")

viz_test_objects_embedding(transfer_class=myclass, Encoder=Encoder, Classifier=Classifier,
                           test_accuracy=accuracy, pred_label_target=pred_label_target)

# %% Parameters tuning
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
