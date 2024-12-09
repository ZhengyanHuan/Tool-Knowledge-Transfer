import torch

behavior_list = ['3-stirring-fast']
modality_list = ['audio']
source_tool_list = ['plastic-spoon', ]  # only one source tool
assist_tool_list = ['wooden-fork', 'metal-whisk', "wooden-chopstick", "plastic-knife"]  # can be treated as source tool
target_tool_list = ['metal-scissor']  # only one target tool
all_tool_list = source_tool_list + assist_tool_list + target_tool_list
trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
enc_trial_list = [0, 1, 2, 3, 4, 5, 6, 7]  # for baseline1, use the remaining trials to test classifier
trial_val_portion = 0.2
randomize_trials = True

old_object_list = ['cane-sugar', 'chia-seed', 'empty', 'glass-bead', 'plastic-bead',
                   'salt', 'kidney-bean', 'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['chickpea', 'detergent', 'metal-nut-bolt', 'split-green-pea', 'wheat']
all_object_list = old_object_list + new_object_list
loss_func = "sincere"  # "TL" for triplet loss or "sincere"
data_name = 'audio_16kHz_token_down16_beh3.bin'  # downsized and flattened token vectors from behavior 3, len=744
# data_name =  "dataset_discretized.bin"

enc_pt_folder = './saved_model/encoder/'
encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_folder = './saved_model/classifier/'
clf_pt_name = f"myclassifier_{loss_func}.pt"

##### context selection for experiments
encoder_exp_name = "assist"  # besides source, use old object from assist tool to train encoder.
# encoder_exp_name = "assist_no-target"  #  use old object from assist tool to train encoder but no target tool for encoder
# encoder_exp_name = "baseline1"   # no transfer, train on target tool and test on target tool
# encoder_exp_name = "baseline2"   # no transfer, train on source tool(s) and test on target tool
# encoder_exp_name = "baseline2-all"  # no transfer, train on all other tools and test on target tool
# encoder_exp_name = "default"  # source to target transfer
# encoder_exp_name = "all"  # use all other tools (source+assist) to train encoder

clf_exp_name = "assist"  # besides source, use assist tool to train clf on new objects.
# clf_exp_name = "default"  # use source tool to train clf on new objects.

exp_pred_obj = "new"  # default, classifier only predicts new object
# exp_pred_obj = "all"   # classifier predicts all object

####### options by main.py running order
retrain_encoder = True
retrain_clf = True
save_temp_model = True

# viz:
viz_dataset = True
viz_share_space = True
viz_decision_boundary = True
plot_learning = True
save_fig = True
viz_l2_norm = False  # viz l2 normed data in 2d space

########## for cross validation ##########
cross_validate = False
lr_encoder = 1e-3  # encoder lr
TL_margin = 0.5  # TL alpha

########## default hyper params ##########
# total epoch patience for encoder is early_stop_patience_enc * smooth_wind_size epochs
early_stop_patience_enc = 2  # None or int
smooth_wind_size = 10  # check progression in every <smooth_wind_size> epochs

early_stop_patience_clf = 100  # None or int
tolerance = 5e-3
clf_tolerance = 1e-4

# encoder parameters
encoder_hidden_dim = 256
encoder_output_dim = 128  # 2 makes it easy to visualize decision boundary
epoch_encoder = 2000

# TL loss parameters
pairs_per_batch_per_object = 300   # smaller value fluctuates the loss

# SINCERE loss parameters
sincere_temp = 0.5

# classifier parameters
epoch_classifier = 5000    # we have early stopping now
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


