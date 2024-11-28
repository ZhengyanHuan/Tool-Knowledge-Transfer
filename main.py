import os
import random
import time

import numpy as np
import torch

import configs
import model
from data.helpers import viz_input_data, viz_embeddings
from transfer_class import Tool_Knowledge_transfer_class

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# for reproducibility
seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU.

# start_time = time.time()

behavior_list = ['3-stirring-fast']
source_tool_list = ['plastic-spoon']  # ['plastic-spoon','wooden-fork', 'metal-whisk']
target_tool_list = ['metal-scissor']
modality_list = ['audio']
trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

old_object_list = ['cane-sugar', 'chia-seed', 'empty', 'glass-bead', 'kidney-bean', 'salt', 'split-green-pea',
                   'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['chickpea', 'detergent', 'metal-nut-bolt', 'plastic-bead', 'wheat']
loss_func = "sincere"  # "TL" for triplet loss or "sincere"
data_name = 'audio_16kHz_token_down16_beh3.bin'
# data_name =  "dataset_discretized.bin"
print(f"input data name: {data_name}")
myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func, data_name=data_name)

input_dim = 0
for modality in modality_list:
    input_dim += myclass.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][
        0].__len__()

encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_name = f"myclassifier_{loss_func}.pt"
retrain_encoder = True
retrain_clr = True
start_time = time.time()

print("👀 visualize initial data ...")
viz_input_data(data=myclass.data_dict, loss_func_name=loss_func, behavior_list=behavior_list,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               old_object_list=old_object_list, new_object_list=new_object_list, shared_only=False, test_only=False)
print("👀 visualize initial data from test objects only ...")
viz_input_data(data=myclass.data_dict, loss_func_name=loss_func, behavior_list=behavior_list,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               plot_title="Original Data Space-Test Only",
               old_object_list=old_object_list, new_object_list=new_object_list, shared_only=False, test_only=True)

if retrain_encoder:
    print(f"👉training representation encoder using {loss_func} loss...")
    encoder_time = time.time()
    myencoder = myclass.train_encoder(
        behavior_list=behavior_list, source_tool_list=source_tool_list, target_tool_list=target_tool_list,
        old_object_list=old_object_list, new_object_list=new_object_list, modality_list=modality_list,
        trail_list=trail_list)
    torch.save(myencoder.state_dict(), './saved_model/encoder/' + encoder_pt_name)
    print(
        f"⏱️Time used for encoder training: {round((time.time() - encoder_time) // 60)} min {(time.time() - encoder_time) % 60:.1f} sec.")

print("👀visualize the shared object set in shared latent space...")
viz_embeddings(viz_objects="shared", loss_func=loss_func, input_dim=input_dim,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               modality_list=modality_list, trail_list=trail_list, behavior_list=behavior_list,
               old_object_list=old_object_list, new_object_list=new_object_list, transfer_class=myclass)

print("👀visualize the test object set in shared latent space...")
viz_embeddings(viz_objects="test", loss_func=loss_func, input_dim=input_dim,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               modality_list=modality_list, trail_list=trail_list, behavior_list=behavior_list,
               old_object_list=old_object_list, new_object_list=new_object_list, transfer_class=myclass)

print(f"👀visualize all embeddings in  in shared latent space...")
viz_embeddings(viz_objects="all", loss_func=loss_func, input_dim=input_dim,
               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
               modality_list=modality_list, trail_list=trail_list, behavior_list=behavior_list,
               old_object_list=old_object_list, new_object_list=new_object_list, transfer_class=myclass)

# %%
if retrain_clr:
    print(f"👉training classification head...")
    clf_time = time.time()

    Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                            hidden_size=configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(
        torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(
        behavior_list=behavior_list, source_tool_list=source_tool_list, new_object_list=new_object_list,
        modality_list=modality_list, trail_list=trail_list, Encoder=Encoder)
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

    print(
        f"⏱️Time used for classifier training: {round((time.time() - clf_time) // 60)} min {(time.time() - clf_time) % 60:.1f} sec.")

# %%
Encoder = model.encoder(input_size=input_dim, output_size=configs.encoder_output_dim,
                        hidden_size=configs.encoder_hidden_dim).to(configs.device)
Encoder.load_state_dict(
    torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))

Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)
Classifier.load_state_dict(
    torch.load('./saved_model/classifier/' + clf_pt_name, map_location=torch.device(configs.device)))

print(f"Evaluating the classifier...")
myclass.eval(Encoder, Classifier, behavior_list, target_tool_list, new_object_list, modality_list, trail_list)
print(f"⏱️total time used: {round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")

# #%% Parameters tuning
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
#     input_dim+=myclass.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0].__len__()
#
#
# #%%
# number_of_folds = 4
# alpha_list = [0.5,1]
# lr_en_list = [0.01,0.1]
#
# best_alpha, best_lr_en = train.train_TL_k_fold(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, number_of_folds, alpha_list, lr_en_list)
# test_acc = train.train_TL_fixed_para(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, best_alpha, best_lr_en)
