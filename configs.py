import random

import numpy as np
import torch

behavior_list = ['3-stirring-fast']
modality_list = ['audio']
source_tool_list = ['plastic-spoon', ]  # only one source tool
assist_tool_list = ['wooden-fork', 'metal-whisk', "wooden-chopstick", "plastic-knife"]  # can be treated as source tool
target_tool_list = ['metal-scissor']  # only one target tool
all_tool_list = source_tool_list + assist_tool_list + target_tool_list
trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
enc_trail_list = [0, 1, 2, 3, 4, 5, 6, 7]  # used for baseline
trial_val_portion = 0.2
randomize_trials = True

old_object_list = ['cane-sugar', 'chia-seed', 'empty', 'glass-bead', 'plastic-bead',
                   'salt', 'kidney-bean', 'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['chickpea', 'detergent', 'metal-nut-bolt', 'split-green-pea', 'wheat']
all_object_list = old_object_list + new_object_list
loss_func = "sincere"  # "TL" for triplet loss or "sincere"
data_name = 'audio_16kHz_token_down16_beh3.bin'  # downsized and flattened token vectors from behavior 3, len=744
# data_name =  "dataset_discretized.bin"

encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_name = f"myclassifier_{loss_func}.pt"

##### context selection for experiments
encoder_exp_name = "assist"  # besides source, use old object from assist tool to train encoder. default is ""
# encoder_exp_name = "assist_no-target"  #  # no target tool for encoder, but project all source and assist tools to train clf
# encoder_exp_name = "baseline1"
# encoder_exp_name = "baseline2"
# encoder_exp_name = "baseline2-all"
# encoder_exp_name = "default"
# encoder_exp_name = "all"  # use all other tools (source+assist) to train encoder

clf_exp_name = "assist"  # besides source, use new object from assist tool to train clf. default is ""
# clf_exp_name = "default"

exp_pred_obj = "new"  # default, classifier only predicts new object
# exp_pred_obj = "all"

####### options by main.py running order
viz_dataset = False
retrain_encoder = False
viz_share_space = False
retrain_clr = False
viz_share_space_l2_norm = False

########## for cross validation ##########
cross_validate = False
lr_encoder = 1e-3  # encoder lr
TL_margin = 1  # TL alpha

########## default hyper params ##########
# total patience is early_stop_patience*smooth_wind_size epochs
early_stop_patience = 3
smooth_wind_size = 50  # check progression in every 50 epochs
tolerance = 1e-3

# encoder parameters
encoder_hidden_dim = 32
encoder_output_dim = 16  # 2 makes it easy to visualize decision boundary
epoch_encoder = 2000

# TL loss parameters
pairs_per_batch_per_object = 10

# SINCERE loss parameters
sincere_temp = 0.5

# classifier parameters
epoch_classifier = 3000
lr_classifier = 1e-2

#################################################
train_category_num = 9
val_category_num = 3
test_category_num = 3

new_object_num = 3

#########################################
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # apple M1 chip
else:
    device = 'cpu'

rand_seed = 43


def set_torch_seed(seed=rand_seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)

