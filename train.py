import logging
import time
import warnings
from typing import List, Union

import torch

import configs
import model
from my_helpers.viz_helpers import viz_test_objects_embedding
from transfer_class import Tool_Knowledge_transfer_class

encoder_pt_name = f"tmp_myencoder.pt"
clf_pt_name = f"tmp_myclassifier.pt"


def train_TL_k_fold(myclass: Tool_Knowledge_transfer_class, train_val_list: List[str],
                    number_of_folds: int, alpha_list: List[Union[float, int]], lr_en_list: List[Union[float, int]],
                    source_tool_list: List[str], target_tool_list: List[str],
                    behavior_list=configs.behavior_list, modality_list=configs.modality_list,
                    trail_list=configs.trail_list, plot_learning=False, enc_trial_split=0.2):
    '''
    cross validation split on objects: some objects for train, the rest objects for val
    Assume that the last 3 objects are unknown to the tool, we use the 12 known ones to split train and val.
    '''
    # train_val_list = all_object_list[:12]
    # test_list = all_object_list[12:]
    best_acc = -1
    best_alpha = -1
    best_lr_en = -1
    rand_guess_acc = -1
    curr_fold_len = len(train_val_list) // number_of_folds

    logging.info("###########################search start###########################")
    for alpha in alpha_list:
        for lr_en in lr_en_list:
            logging.info("Learning rate for the encoder is:  " + str(lr_en))
            logging.info("TL margin is:  " + str(alpha))
            logging.info("=========================cv start=========================")
            # start cv for current hyper-param combo
            acc_sum = 0
            cv_start_time = time.time()
            for fold_idx in range(number_of_folds):
                logging.info(f"------------------fold {fold_idx + 1}/{number_of_folds} start-----------------")
                obj_fold_len = len(train_val_list) // number_of_folds
                if fold_idx == number_of_folds - 1 and (fold_idx + 1) * obj_fold_len < len(train_val_list):
                    end_idx = len(train_val_list)  # make sure the last val fold covers the rest of objects
                    curr_fold_len = len(train_val_list) - fold_idx * obj_fold_len
                else:
                    end_idx = (fold_idx + 1) * obj_fold_len
                val_list = train_val_list[fold_idx * obj_fold_len: end_idx]
                train_list = [item for item in train_val_list if item not in val_list]
                ######
                logging.info(f"👉 training representation encoder...")
                encoder_time = time.time()
                configs.set_torch_seed()
                Encoder = myclass.train_encoder(
                    lr_en=lr_en, TL_margin=alpha, behavior_list=behavior_list, source_tool_list=source_tool_list,
                    target_tool_list=target_tool_list, old_object_list=train_list, new_object_list=val_list,
                    modality_list=modality_list, trail_list=trail_list, plot_learning=plot_learning,
                    early_stop_patience=50, trial_split=enc_trial_split)
                # old list is all the list except val_list+test_list, i.e., train_list TODO: DISCUSS
                torch.save(Encoder.state_dict(), './saved_model/encoder/' + encoder_pt_name)
                logging.info(f"Time used for encoder training: {round((time.time() - encoder_time) // 60)} min "
                             f"{(time.time() - encoder_time) % 60:.1f} sec.")

                ########
                logging.info(f"👉 training classification head...")
                clf_time = time.time()
                # Encoder = model.encoder(input_size=input_dim).to(configs.device)
                # Encoder.load_state_dict(torch.load('./saved_model/encoder/' + encoder_pt_name,
                #                                    map_location=torch.device(configs.device)))
                configs.set_torch_seed()
                Classifier = myclass.train_classifier(Encoder=Encoder, behavior_list=behavior_list,
                                                      source_tool_list=source_tool_list, new_object_list=val_list,
                                                      modality_list=modality_list, trail_list=trail_list,
                                                      plot_learning=plot_learning)
                torch.save(Classifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

                logging.info(f"Time used for classifier training: {round((time.time() - clf_time) // 60)} min "
                             f"{(time.time() - clf_time) % 60:.1f} sec.")

                ##########
                # Encoder = model.encoder(input_size=input_dim).to(configs.device)
                # Encoder.load_state_dict(torch.load('./saved_model/encoder/' + encoder_pt_name,
                #                                    map_location=torch.device(configs.device)))
                #
                # Classifier = model.classifier(output_size=len(val_list)).to(configs.device)
                # Classifier.load_state_dict(torch.load('./saved_model/classifier/' + clf_pt_name,
                #                                       map_location=torch.device(configs.device)))

                logging.info(f"👉 Evaluating the classifier...")
                val_acc = myclass.eval(Encoder=Encoder, Classifier=Classifier, behavior_list=behavior_list,
                                       tool_list=target_tool_list, new_object_list=val_list,
                                       modality_list=modality_list, trail_list=trail_list)
                acc_sum += val_acc
                logging.info(f"👉 fold {fold_idx + 1}/{number_of_folds} val_obj: {val_list} \n"
                             f"TL margin: {alpha}, lr: {lr_en}, test accuracy: {val_acc * 100:.2f}%")

            logging.info(f"☑️ total time used for this cv: {round((time.time() - cv_start_time) // 60)} min "
                         f"{(time.time() - cv_start_time) % 60:.1f} sec.")

            avg_acc = acc_sum / number_of_folds
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_alpha = alpha
                best_lr_en = lr_en
                rand_guess_acc = 1 / curr_fold_len
    logging.info(f"✅ The best avg val accuracy is: {best_acc * 100:.1f}%, "
                 f"random guess accuracy: {rand_guess_acc * 100:.2f}%")
    return best_alpha, best_lr_en


def train_TL_fixed_param(myclass: Tool_Knowledge_transfer_class, train_val_obj_list: List[str], test_obj_list: List[str],
                         input_dim: int, best_alpha: float, best_lr_en: Union[float, int], best_lr_clf=configs.lr_classifier,
                         behavior_list=configs.behavior_list, source_tool_list=configs.source_tool_list,
                         target_tool_list=configs.target_tool_list, test_name="test_fold_",
                         modality_list=configs.modality_list, trail_list=configs.trail_list):
    logging.warning("train_TL_fixed_param function currently only applies the best TL alpha and training learning rate"
                    ", all other hyper-params are by default from configs.")
    start_time = time.time()
    Encoder = myclass.train_encoder(
        lr_en=best_lr_en, TL_margin=best_alpha,
        source_tool_list=source_tool_list, target_tool_list=target_tool_list,
        old_object_list=train_val_obj_list, new_object_list=test_obj_list,
        behavior_list=behavior_list, modality_list=modality_list, trail_list=trail_list)
    torch.save(Encoder.state_dict(), './saved_model/encoder/' + test_name + encoder_pt_name)

    Classifier = myclass.train_classifier(
        Encoder=Encoder, lr_clf=best_lr_clf, source_tool_list=source_tool_list, trail_list=trail_list,
        new_object_list=test_obj_list, behavior_list=behavior_list, modality_list=modality_list)
    torch.save(Classifier.state_dict(), './saved_model/classifier/' + test_name + clf_pt_name)
    #
    # Encoder = model.encoder(input_size=input_dim).to(configs.device)
    # Encoder.load_state_dict(torch.load('./saved_model/encoder/' + test_name + encoder_pt_name,
    #                                    map_location=torch.device(configs.device)))
    #
    # Classifier = model.classifier(output_size=len(test_obj_list)).to(configs.device)
    # Classifier.load_state_dict(torch.load('./saved_model/classifier/' + test_name + clf_pt_name,
    #                                       map_location=torch.device(configs.device)))

    logging.info(f"Evaluating the classifier...")
    test_acc, _, pred_label_target = myclass.eval(Encoder=Encoder, Classifier=Classifier, return_pred=True,
                                                  new_object_list=test_obj_list, tool_list=target_tool_list)

    logging.info(f"✅✅✅ test accuracy is: {test_acc * 100:.1f}%, "
                 f"random guess accuracy: {100/len(test_obj_list):.2f}%")

    viz_test_objects_embedding(transfer_class=myclass, Encoder=myclass.trained_encoder,
                               Classifier=myclass.trained_clf, new_object_list=test_obj_list,
                               source_tool_list=source_tool_list, target_tool_list=target_tool_list,
                               assist_tool_list=[], test_accuracy=test_acc, pred_label_target=pred_label_target)

    logging.info(f"☑️ total time used for refit and test: {round((time.time() - start_time) // 60)} min "
                 f"{(time.time() - start_time) % 60:.1f} sec.")

    return test_acc
