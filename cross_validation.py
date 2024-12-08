import json
import random
import sys
import time
import logging
import os

import numpy as np

import configs
import train
from my_helpers.data_helpers import select_context_for_experiment, fill_missing_params, fill_missing_context
from my_helpers.pipeline import run_pipeline
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

# for reproducibility
configs.set_torch_seed()
# %% 1. task setup
main_logger.debug(f"========================= New Run =========================")  # new log starts here
data_name = 'audio_16kHz_token_down16_beh3.bin'
loss_func = "sincere"  # 👈
exp_name = "baseline1"  # 👈
exp_pred_obj = "all"  # 👈
pipe_settings = {'encoder_exp_name': exp_name, "exp_pred_obj": exp_pred_obj,
                 'clf_exp_name': "assist" if "assist" in exp_name else "default"}

all_obj_list = sorted(configs.all_object_list)

# %% setup
# outside test
test_size = 5
num_test_fold = 1  # 👈

# inside CV
cross_validate = True
number_of_cv_folds = 1  # 👈 for hyper-param tuning
no_overlap_sample = False
plot_learning = True
pipe_settings['plot_learning'] = plot_learning


alpha_list = [0.5, 1]  # parameter for triplet Loss: TL margin
lr_en_list = [0.001, 0.01]  # learning rate for encoder
grid = {"alpha_list": alpha_list if loss_func == "TL" else [None],  # only search for TL
        "lr_en_list": lr_en_list}  # 👈add params to tune
main_logger.info(f"search grid for {loss_func} loss: {grid}")

# %% test tracker
test_result_dict = {f"{loss_func}": {}}
top_folder_name = (f"test_result/source_{configs.source_tool_list[0]}/"
                   f"target_{configs.target_tool_list[0]}/{loss_func}/{exp_pred_obj}")
if not os.path.exists(top_folder_name):
    os.makedirs(top_folder_name)
exp_file_path = os.path.join(top_folder_name, f"test_result_{exp_name}.json")
exp_dict = {exp_name: {}}
# %% start
start_time = time.time()
for i in range(num_test_fold):
    test_fold_name = f"fold{i}"
    random.seed(configs.rand_seed + i)
    test_obj_list = random.sample(all_obj_list, test_size)
    test_result_dict[f"{loss_func}"].update({test_fold_name: {"test_obj_list": test_obj_list}})
    train_val_obj_list = [item for item in all_obj_list if item not in test_obj_list]
    main_logger.info(f"test sample {i + 1}, test_obj_list: {test_obj_list}")

    cv_result = train.train_k_fold(train_val_obj_list=train_val_obj_list, number_of_folds=number_of_cv_folds,
                                   loss_func=loss_func, data_name=data_name, grid=grid,
                                   no_overlap_sample=no_overlap_sample, pipe_settings=pipe_settings)

    test_result_dict[f"{loss_func}"][test_fold_name].update(cv_result)
    # Dump the dictionary as a JSON file
    with open(f"{top_folder_name}/test_result_{exp_name}_temp.json", "w") as json_file:
        json.dump(test_result_dict, json_file, indent=2)

    # %% Test
    main_logger.info(f"search for current test set is Done. start testing...")
    hyparams = {"TL_margin": cv_result['best_alpha'], 'lr_encoder': cv_result['best_lr_en']}  # 👈add tuned params
    test_accuracy = train.train_fixed_param(train_val_obj_list=train_val_obj_list, test_obj_list=test_obj_list,
                                            loss_func=loss_func, data_name=data_name,
                                            hyparams=hyparams, pipe_settings=pipe_settings, test_name=test_fold_name)

    main_logger.info(f"test fold {i + 1}, test_accuracy: {test_accuracy * 100:.1f}%")
    test_result_dict[f"{loss_func}"][test_fold_name].update({"test_accuracy": test_accuracy})

main_logger.info(f"✅ total time used for {num_test_fold} test * {number_of_cv_folds} val folds: "
                 f"{round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")
# %% summarize all test result
accuracy_list = [fold_result['test_accuracy'] for fold_result in test_result_dict[f"{loss_func}"].values()]
hyparams = fill_missing_params({}, param_model="both")
context = fill_missing_context({})
exp_dict[exp_name] = {
    loss_func: {
        "result": {
            "avg_accuracy": np.mean(accuracy_list),
            "std_accuracy": np.std(accuracy_list),
            "accuracy_list": accuracy_list,
            "source_tool": configs.source_tool_list,
            "target_tool": configs.target_tool_list,
            "experiment":
                {"rand_seed": configs.rand_seed,
                 "num_test_folds": num_test_fold,
                 "num_cv_folds_in_test": number_of_cv_folds,
                 "encoder_exp_name": configs.encoder_exp_name,
                 "clf_exp_name": configs.clf_exp_name,
                 "exp_pred_obj": configs.exp_pred_obj
                 },
            "test_result_dict": test_result_dict,
        },
        "other_info": {
            "other_params": hyparams,
            "data_name": data_name,
            "original_context": context,
        }

    }
}

try:
    with open(exp_file_path, "w") as json_file:
        json.dump(exp_dict, json_file, indent=2)
    os.makedirs(f"{top_folder_name}/test_result_{exp_name}_temp.json")
finally:
    main_logger.info(f"Done.")


