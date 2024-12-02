import torch

behavior_list = ['3-stirring-fast']
modality_list = ['audio']
# source_tool_list = ['plastic-spoon']
source_tool_list = ['plastic-spoon', 'wooden-fork', 'metal-whisk', 'plastic-knife', 'wooden-chopstick']
target_tool_list = ['metal-scissor']
all_tool_list = source_tool_list+target_tool_list
trail_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

old_object_list = ['cane-sugar', 'chia-seed', 'empty', 'glass-bead', 'plastic-bead',
                   'salt', 'kidney-bean', 'styrofoam-bead', 'water', 'wooden-button']
new_object_list = ['chickpea', 'detergent', 'metal-nut-bolt', 'split-green-pea', 'wheat']
all_object_list = old_object_list+new_object_list
loss_func = "TL"  # "TL" for triplet loss or "sincere"
data_name = 'audio_16kHz_token_down16_beh3.bin'  # downsized and flattened token vectors from behavior 3, len=744
# data_name =  "dataset_discretized.bin"

encoder_pt_name = f"myencoder_{loss_func}.pt"
clf_pt_name = f"myclassifier_{loss_func}.pt"
retrain_encoder = True
retrain_clr = True

encoder_exp_name = "source-assist"  # besides source, use old object from assist tool to train encoder. default is ""
# encoder_exp_name = "source-all":  # use all other tools (source+assist) to train encoder
clf_exp_name = "source-assist"  # besides source, use new object from assist tool to train clf. default is ""
# encoder_exp_name = ""
# clf_exp_name = ""
if "assist" not in encoder_exp_name:
    if "all" in encoder_exp_name:
        source_tool_list += assist_tool_list
    assist_tool_list = []
viz_dataset = True
viz_process = True

#########################encoder parameters
l2_norm = loss_func == "sincere"

encoder_hidden_dim = 256
encoder_output_dim = 4  # 2 makes it easy to visualize decision boundary

epoch_encoder = 2000
lr_encoder = 1e-3
############## TL loss parameters ##########
TL_margin = 1
pairs_per_batch_per_object = 10

############## SINCERE loss parameters ##########
sincere_temp = 0.1

#############classifier parameters #######
epoch_classifier = 2000
lr_classifier = 1e-1
val_portion = 0

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

def set_torch_seed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)
