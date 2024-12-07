import time
import logging
import os

import configs
import train
from my_helpers.data_helpers import select_context_for_experiment
from my_helpers.viz_helpers import viz_test_objects_embedding
from transfer_class import Tool_Knowledge_transfer_class

# %% 0. script setup
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
logging.basicConfig(level=logging.DEBUG, filename=log_file_path + "/log_file_cv.log",
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
 clf_source_tool_list, clf_target_tool_list) = select_context_for_experiment(configs.encoder_exp_name,
                                                                             configs.clf_exp_name)

input_dim = 0
data_dim = 0
for modality in configs.modality_list:
    obj_trial_batch = myclass.data_dict[configs.behavior_list[0]][configs.target_tool_list[0]][modality]
    x_sample = obj_trial_batch[configs.old_object_list[0]]['X'][0]
    input_dim += len(x_sample)
    data_dim = x_sample.shape[-1]




# %%
# train_val_list = ['detergent', 'kidney-bean', 'plastic-bead', 'chia-seed', 'salt', 'empty', 'metal-nut-bolt',
#                   'wooden-button', 'styrofoam-bead', 'water', 'glass-bead', 'wheat']
# test_list = ['cane-sugar', 'split-green-pea', 'chickpea']
# todo: randomize list

source_tool_list = ['plastic-spoon']
target_tool_list = ['metal-scissor']
loss_func = "TL"  # "TL" for triplet loss or "sincere"
myclass = Tool_Knowledge_transfer_class(encoder_loss_fuc=loss_func)

input_dim = 0
for modality in configs.modality_list:
    input_dim += len(myclass.data_dict[configs.behavior_list[0]][configs.target_tool_list[0]
                     ][modality][test_list[0]]['X'][0])

start_time = time.time()
# %%
number_of_folds = 4  # num of folds for cross validation on train_val_list. e.g, 12 obj: 3 val, 9 train
alpha_list = [0.5, 1]  # parameter for triplet Loss: margin
lr_en_list = [0.01, 0.1]  # learning rate for encoder and classifier

if configs.cross_validate is True:
    best_alpha, best_lr_en = train.train_TL_k_fold(
        myclass=myclass, train_val_list=train_val_list, number_of_folds=number_of_folds,
        lr_en_list=lr_en_list, alpha_list=alpha_list,
        source_tool_list=source_tool_list, target_tool_list=target_tool_list, plot_learning=False)
else:
    best_alpha, best_lr_en = configs.TL_margin, configs.lr_encoder
test_acc = train.train_TL_fixed_param(
    myclass=myclass, train_val_obj_list=train_val_list, test_obj_list=test_list,
    source_tool_list=source_tool_list, target_tool_list=target_tool_list,
    input_dim=input_dim, best_alpha=best_alpha, best_lr_en=best_lr_en, test_name="test_fold0_")
