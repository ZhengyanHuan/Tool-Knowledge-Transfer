import torch
import time
import model
import configs

def train_TL_k_fold(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, number_of_folds, alpha_list, lr_en_list):
    '''
    Assume that the last 3 objects are unknow to the tool, we use the 12 known ones to split train and val.
    '''
    # train_val_list = all_object_list[:12]
    # test_list = all_object_list[12:]
    encoder_pt_name = f"myencoder_tmp.pt"
    clf_pt_name = f"myclassifier_tmp.pt"

    best_acc = -1
    best_alpha = -1
    best_lr_en = -1
    for alpha in alpha_list:

        for lr_en in lr_en_list:
            print("Learning rate for the encoder is:  " +str(lr_en))
            print("TL margin is:  " + str(alpha))
            for i in range(number_of_folds):
                acc_sum = 0
                fold_len = len(train_val_list )//number_of_folds
                val_list = train_val_list[ i *fold_len: ( i +1 ) *fold_len]
                train_list = [item for item in train_val_list if item not in val_list]

                ######
                print(f"training representation encoder...")
                encoder_time = time.time()
                myencoder = myclass.train_encoder(behavior_list, source_tool_list, target_tool_list, val_list +test_list,
                                                  modality_list, trail_list) # old list is all the list except val_list+test_list, i.e., train_list
                torch.save(myencoder.state_dict(), './saved_model/encoder/' + encoder_pt_name)
                print(
                    f"Time used for encoder training: {round((time.time() - encoder_time) // 60)} min {(time.time() - encoder_time) % 60:.1f} sec.")

                ########

                print(f"training classification head...")
                clf_time = time.time()

                Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(
                    configs.device)
                Encoder.load_state_dict(
                    torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))
                myclassifier = myclass.train_classifier(behavior_list, source_tool_list, val_list, modality_list,
                                                        trail_list, Encoder)
                torch.save(myclassifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

                print(
                    f"Time used for classifier training: {round((time.time() - clf_time) // 60)} min {(time.time() - clf_time) % 60:.1f} sec.")

                ##########
                start_time = time.time()
                Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(
                    configs.device)
                Encoder.load_state_dict(
                    torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))

                Classifier = model.classifier(configs.encoder_output_dim, len(val_list)).to(configs.device)
                Classifier.load_state_dict(
                    torch.load('./saved_model/classifier/' + clf_pt_name, map_location=torch.device(configs.device)))

                print(f"Evaluating the classifier...")
                val_acc = myclass.eval(Encoder, Classifier, behavior_list, target_tool_list, val_list, modality_list,
                                       trail_list)
                acc_sum += val_acc

                print(
                    f"total time used: {round((time.time() - start_time) // 60)} min {(time.time() - start_time) % 60:.1f} sec.")

            avg_acc = acc_sum /number_of_folds
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_alpha = alpha
                best_lr_en = lr_en
    print("The best avg val accuracy is: " + str(best_acc.item()))
    return best_alpha, best_lr_en

def train_TL_fixed_para(myclass, train_val_list, test_list, behavior_list ,source_tool_list, target_tool_list, modality_list ,trail_list ,input_dim, alpha, lr_en):
    encoder_pt_name = f"myencoder_best_para.pt"
    clf_pt_name = f"myclassifier_best_para.pt"

    ######
    print(f"training representation encoder...")
    encoder_time = time.time()
    myencoder = myclass.train_encoder(behavior_list, source_tool_list, target_tool_list,  test_list,
                                      modality_list,
                                      trail_list)  # old list is all the list except val_list+test_list, i.e., train_list
    torch.save(myencoder.state_dict(), './saved_model/encoder/' + encoder_pt_name)
    print(
        f"Time used for encoder training: {round((time.time() - encoder_time) // 60)} min {(time.time() - encoder_time) % 60:.1f} sec.")

    ########

    print(f"training classification head...")
    clf_time = time.time()

    Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(
        configs.device)
    Encoder.load_state_dict(
        torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(behavior_list, source_tool_list, test_list, modality_list,
                                            trail_list, Encoder)
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

    print(
        f"Time used for classifier training: {round((time.time() - clf_time) // 60)} min {(time.time() - clf_time) % 60:.1f} sec.")

    ##########
    start_time = time.time()
    Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(
        configs.device)
    Encoder.load_state_dict(
        torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))

    Classifier = model.classifier(configs.encoder_output_dim, len(test_list)).to(configs.device)
    Classifier.load_state_dict(
        torch.load('./saved_model/classifier/' + clf_pt_name, map_location=torch.device(configs.device)))

    print(f"Evaluating the classifier...")
    test_acc = myclass.eval(Encoder, Classifier, behavior_list, target_tool_list, test_list, modality_list,
                            trail_list)

    return test_acc