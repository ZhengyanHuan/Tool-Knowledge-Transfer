# %%
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import configs
import model
from data.helpers import sanity_check_data_labels, SORTED_DATA_OBJ_LIST
from sincere_loss_class import SINCERELoss


# %%
class Tool_Knowledge_transfer_class:
    def __init__(self, encoder_loss_fuc="TL", data_name="dataset_discretized.bin"):
        """
        :param encoder_loss_fuc: "TL" for triplet loss or "sincere"
        :param: data_name: "audio_16kHz_token_down16_beh3.bin" for behavior3 only, tokenized audio by BEATS model,
                            downsized by 16 and flattened to 1D.
                           "dataset_discretized.bin" for discretized data, flattened to 1D
        """

        ####load dataset
        robots_data_filepath = r'data' + os.sep + data_name
        bin_file = open(robots_data_filepath, 'rb')
        robot = pickle.load(bin_file)
        bin_file.close()

        self.data_dict = robot
        sanity_check_data_labels(self.data_dict)
        self.encoder_loss_fuc = encoder_loss_fuc

        #### load names
        data_file_path = os.sep.join([r'data', 'dataset_metadata.bin'])
        bin_file = open(data_file_path, 'rb')
        metadata = pickle.load(bin_file)
        bin_file.close()

        self.behaviors = list(metadata.keys())
        self.objects = metadata[self.behaviors[0]]['objects']
        self.tools = metadata[self.behaviors[0]]['tools']
        self.trials = metadata[self.behaviors[0]]['trials']
        logging.debug(f"behaviors:, {len(self.behaviors)}, {self.behaviors}")
        logging.debug(f"objects: , {len(self.objects)}, {self.objects}")
        logging.debug(f"tools: , {len(self.tools)}, {self.tools}")
        logging.debug(f'trials: , {len(self.trials)}, {self.trials}')

        ####
        self.CEloss = torch.nn.CrossEntropyLoss()
        # self.Classifier = model.classifier(configs.encoder_output_dim, configs.new_object_num)

    def plot_func(self, record, type, save_name='test', plot_every=10):  # type-> 'encoder', 'classifier'
        logging.debug(f"➡️ plot_func for {type}: {save_name}...")
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 18

        color_group = ['red', 'blue']
        if type == 'encoder':
            encoder_param = configs.TL_margin if configs.loss_func == "TL" else configs.sincere_temp
            encoder_param_name = "margin" if configs.loss_func == "TL" else "temperature"
            xaxis = np.arange(1, len(record) + 1)
            plt.plot(xaxis[::plot_every], record[::plot_every], color_group[0])
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title(f'Encoder Training Loss Progression - Loss: {configs.loss_func} \n '
                      f'Epochs: lr: {configs.lr_encoder}, {encoder_param_name}: {encoder_param}'
                      f', emb_size: {configs.encoder_output_dim}')
            plt.grid()
            plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
            plt.show()
            plt.close()
        elif type == 'classifier':
            xaxis = np.arange(1, record.shape[1] + 1)
            plt.plot(xaxis[::plot_every], record[0, ::plot_every], color_group[0], label='tr loss')
            if record[1, -1] != 0:
                plt.plot(xaxis[::plot_every], record[1, ::plot_every], color_group[1], label='val loss')
            plt.xlabel('epochs')
            plt.title(f'Classifier Training Loss Progression \n '
                      f'Epochs: lr: {configs.lr_classifier}')
            plt.ylabel('loss')
            plt.grid()
            plt.legend()
            plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            logging.warning(f'invalid model type: {type}, plot not available.')

    def prepare_data_classifier(self, behavior_list, source_tool_list, new_object_list, modality_list, trail_list,
                                Encoder):
        logging.debug(f"➡️ prepare_data_classifier..")
        train_test_index = int(len(trail_list) * (1 - configs.val_portion))
        logging.debug(f"get source data for classifier... {new_object_list}")
        source_data = self.get_data(behavior_list, source_tool_list, modality_list, new_object_list, trail_list)
        with torch.no_grad():
            encoded_source = Encoder(source_data)

        train_encoded_source = encoded_source[:, :, :, :train_test_index, :]
        val_encoded_source = encoded_source[:, :, :, train_test_index:, :]

        truth = np.zeros_like(encoded_source[:, :, :, :, -1].cpu())
        # truth = np.zeros([len(new_object_list), len(trail_list)])
        for i in range(len(new_object_list)):
            truth[:, :, i, :] = i
        truth = torch.tensor(truth, dtype=torch.int64, device=configs.device)

        train_truth = truth[:, :train_test_index]
        val_truth = truth[:, train_test_index:]
        logging.debug(f"train_truth: \n      {train_truth}")
        logging.debug(f"val_truth: \n      {val_truth}")

        return train_encoded_source, val_encoded_source, train_truth.reshape(-1), val_truth.reshape(-1)

    def train_classifier(self, Encoder, behavior_list=configs.behavior_list, trail_list=configs.trail_list,
                         new_object_list=configs.new_object_list, modality_list=configs.modality_list,
                         source_tool_list=configs.source_tool_list, lr_clf=configs.lr_classifier):
        loss_record = np.zeros([2, configs.epoch_classifier])
        logging.debug(f"➡️ train_classifier..")

        train_encoded_source, val_encoded_source, train_truth_flat, val_truth_flat = self.prepare_data_classifier(
            behavior_list, source_tool_list, new_object_list, modality_list, trail_list, Encoder)
        Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)

        optimizer = optim.AdamW(Classifier.parameters(), lr=lr_clf)

        for i in range(configs.epoch_classifier):
            pred_tr = Classifier(train_encoded_source)
            pred_flat_tr = pred_tr.view(-1, len(new_object_list))  # (num_data, num_class)
            loss_tr = self.CEloss(pred_flat_tr, train_truth_flat)
            loss_record[0, i] = loss_tr.detach().cpu().numpy()

            if len(val_truth_flat > 0):
                with torch.no_grad():
                    pred_val = Classifier(val_encoded_source)
                    pred_flat_val = pred_val.view(-1, len(new_object_list))
                    loss_val = self.CEloss(pred_flat_val, val_truth_flat)
                    loss_record[1, i] = loss_val.detach().cpu().numpy()

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                pred_label = torch.argmax(pred_flat_tr, dim=-1)
                correct_num = torch.sum(pred_label == train_truth_flat)
                accuracy_train = correct_num / len(train_truth_flat)

                logging.info(f"epoch {i + 1}/{configs.epoch_classifier}, train loss: {loss_tr.item():.4f}, "
                             f"train accuracy: {accuracy_train.item() * 100 :.2f}%")
                if len(val_truth_flat > 0):
                    pred_label = torch.argmax(pred_flat_val, dim=-1)
                    correct_num = torch.sum(pred_label == val_truth_flat)
                    accuracy_val = correct_num / len(val_truth_flat)

                    logging.info(
                        f"epoch {i + 1}/{configs.epoch_classifier}, val loss: {loss_val.item():.4f}, "
                        f"val accuracy: {accuracy_val.item() * 100 :.2f}%")

        self.plot_func(loss_record, 'classifier', f'classifier_{self.encoder_loss_fuc}')
        return Classifier

    def eval(self, Encoder, Classifier, tool_list=configs.target_tool_list, behavior_list=configs.behavior_list,
             new_object_list=configs.new_object_list, modality_list=configs.modality_list,
             trail_list=configs.trail_list, return_pred=False):
        logging.debug(f"➡️ eval..")
        logging.debug(f"{tool_list}: {new_object_list}")
        source_data = self.get_data(behavior_list, tool_list, modality_list, new_object_list, trail_list)
        truth_flat = np.zeros(len(tool_list) * len(trail_list) * len(new_object_list))
        for t in range(len(tool_list)):
            for o in range(len(new_object_list)):
                start = t * (len(new_object_list) * len(trail_list)) + o * len(trail_list)
                truth_flat[start: start + len(trail_list)] = o
        truth_flat = torch.tensor(truth_flat, dtype=torch.int64, device=configs.device)

        with torch.no_grad():
            encoded_source = Encoder(source_data)
            pred = Classifier(encoded_source)
            logging.debug(
                f"source_data: {source_data.shape}, encoded_source: {encoded_source.shape}, pred: {pred.shape}")
        pred_flat = pred.view(-1, len(new_object_list))
        pred_label = torch.argmax(pred_flat, dim=-1)

        logging.info(f"{len(truth_flat)} true labels: {truth_flat.tolist()}")
        logging.info(f"{len(pred_label)} pred labels: {pred_label.tolist()}")
        correct_num = torch.sum(pred_label == truth_flat)
        accuracy_test = correct_num / len(truth_flat)
        logging.info(f"test accuracy: {accuracy_test.item() * 100:.2f}%")
        if return_pred:
            return accuracy_test, pred_flat, pred_label
        else:
            return accuracy_test

    def train_encoder(self, behavior_list=configs.behavior_list,
                      source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
                      old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                      modality_list=configs.modality_list, trail_list=configs.trail_list, lr_en=configs.lr_encoder):
        '''

        :param behavior_list:
        :param source_tool_list:
        :param target_tool_list:
        :param modality_list:
        :param old_object_list: e.g. ['chickpea', 'split-green-pea', 'glass-bead', 'chia-seed', 'wheat',
                                      'wooden-button', 'styrofoam-bead', 'metal-nut-bolt', 'salt']
        :param trail_list: the index of training trails, e.g. [0,1,2,3,4,5,6,7]
        :return:
        '''
        logging.debug(f"➡️ train_encoder..")
        loss_record = np.zeros(configs.epoch_encoder)

        source_data, target_data, truth_source, truth_target = self.get_data_and_convert_labels(
            behavior_list=behavior_list,
            source_tool_list=source_tool_list,
            target_tool_list=target_tool_list,
            modality_list=modality_list,
            old_object_list=old_object_list,
            new_object_list=new_object_list,
            trail_list=trail_list)
        '''
        If we have more than one modality, we may need preprocessing and the input dim may not the 
        sum of data dim across all considered modalities. But I just put it here because we have 
        not figured out what to do.
        '''
        self.input_dim = 0
        for modality in modality_list:
            self.input_dim += len(
                self.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0])

        Encoder = model.encoder(self.input_dim, configs.encoder_output_dim,
                                configs.encoder_hidden_dim).to(configs.device)
        optimizer = optim.AdamW(Encoder.parameters(), lr=lr_en)

        for i in range(configs.epoch_encoder):
            if self.encoder_loss_fuc == "TL":
                loss = self.TL_loss_fn(source_data, target_data, Encoder)
            elif self.encoder_loss_fuc == "sincere":
                loss = self.sincere_ls_fn(source_data, truth_source, target_data, truth_target, Encoder)
            else:
                raise Exception(f"{self.encoder_loss_fuc} not available.")
            loss_record[i] = loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 500 == 0:
                logging.info(f"epoch {i + 1}/{configs.epoch_encoder}, loss: {loss.item():.4f}")

        self.plot_func(loss_record, 'encoder', f'encoder_{self.encoder_loss_fuc}')
        return Encoder

    def sincere_ls_fn(self, source_data, truth_source, target_data, truth_target, Encoder,
                      temperature=configs.sincere_temp):
        all_embeds_norm, all_labels = self.get_embeddings_and_labels(
            Encoder, source_data, target_data, truth_source, truth_target, l2_norm=True)
        sincere_loss = SINCERELoss(temperature)
        if len(all_labels.shape) > 1:
            all_labels = torch.squeeze(all_labels)  # labels (torch.tensor): (B,)
        return sincere_loss(all_embeds_norm, all_labels)

    def get_same_object_list(self, encoded_source, encoded_target):
        same_object_list = []
        tot_len = encoded_source.shape[2]
        target_len = encoded_target.shape[2]
        for i in range(tot_len):
            if i < target_len:
                object_list1 = encoded_source[:, :, i, :, :].reshape([-1, configs.encoder_output_dim])
                object_list2 = encoded_target[:, :, i, :, :].reshape([-1, configs.encoder_output_dim])
                object_list = torch.concat([object_list1, object_list2], dim=0)
            else:
                object_list = encoded_source[:, :, i, :, :].reshape([-1, configs.encoder_output_dim])
            same_object_list.append(object_list)

        return same_object_list

    def TL_loss_fn(self, source_data, target_data, Encoder, alpha=configs.TL_margin):
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)
        same_object_list = self.get_same_object_list(encoded_source, encoded_target)

        trail_tot_num_list = np.array([same_object_list[i].shape[0] for i in range(len(same_object_list))])
        tot_object_num = encoded_source.shape[2]

        A_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)
        P_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)
        N_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)

        for object_index in range(tot_object_num):
            object_list = same_object_list[object_index]

            # Sample anchor and positive
            A_index = np.random.choice(trail_tot_num_list[object_index], size=configs.pairs_per_batch_per_object)
            P_index = np.random.choice(trail_tot_num_list[object_index], size=configs.pairs_per_batch_per_object)
            start = object_index * configs.pairs_per_batch_per_object
            end = (object_index + 1) * configs.pairs_per_batch_per_object
            A_mat[start: end] = object_list[A_index,:]
            P_mat[start: end] = object_list[P_index,:]

            # Sample negative
            N_object_list = np.random.choice(len(trail_tot_num_list), size=configs.pairs_per_batch_per_object)
            N_list = torch.zeros(configs.pairs_per_batch_per_object, configs.encoder_output_dim,
                                 dtype=torch.float32).to(configs.device)
            for i in range(len(N_object_list)):
                N_object_index = N_object_list[i]
                N_trail_index = np.random.choice(trail_tot_num_list[N_object_index])
                N_list[i] = same_object_list[N_object_index][N_trail_index]
                N_mat[start: end] = N_list

        dPA = torch.norm(A_mat - P_mat, dim=1)
        dNA = torch.norm(A_mat - N_mat, dim=1)

        d = dPA - dNA + alpha
        d[d < 0] = 0

        loss = torch.mean(d)
        return loss

    def get_data(self, behavior_list=configs.behavior_list, tool_list=configs.all_tool_list,
                 modality_list=configs.modality_list, object_list=configs.all_object_list,
                 trail_list=configs.trail_list, get_labels=False):
        """
        :return: torch tensor(s). data shape: [n_behaviors, n_tools, n_objects, n_trials, data_dim];
                there's no integer label information, but for each modality, data is ordered by object_list.
                Note that the label index does NOT necessarily start from 0
                    e.g., object_list does NOT start from the old(shared) object_list
        """
        logging.debug(f"➡️get_data...")

        meta_data = {b: {t: {} for t in tool_list} for b in behavior_list}
        if len(modality_list) == 1 and behavior_list and tool_list and object_list and trail_list:
            data_dim = len(self.data_dict[behavior_list[0]][tool_list[0]][modality_list[0]][object_list[0]]['X'][0])
            data = np.zeros((len(behavior_list), len(tool_list), len(object_list), len(trail_list), data_dim))
            label = np.zeros((len(behavior_list), len(tool_list), len(object_list), len(trail_list), 1))
            '''
            Now we have 1 behavior, 1 tool. The data dim is 1x1xtrail_num x data_dim
            But this can work for multiple behaviors, tools
            '''
            for behavior_index, behavior in enumerate(behavior_list):
                for tool_index, tool in enumerate(tool_list):
                    for object_index, object in enumerate(object_list):
                        meta_data[behavior][tool][object] = len(trail_list)
                        for trail_index in range(len(trail_list)):
                            trail = trail_list[trail_index]
                            data[behavior_index][tool_index][object_index][trail_index] = \
                            self.data_dict[behavior][tool][modality_list[0]][object]['X'][trail]
                            label[behavior_index][tool_index][object_index][trail_index] = \
                            self.data_dict[behavior][tool][modality_list[0]][object]['Y'][trail]

            data = torch.tensor(data, dtype=torch.float32, device=configs.device)
            label = torch.tensor(label, dtype=torch.int64, device=configs.device)

        else:
            data = None
            label = None
            '''
            if we have more than one modality, the data dim are different and a tensor cannot hold this.
            So I leave this for future extension.
            '''
        logging.debug(f"get data mata: {meta_data}")
        if get_labels:
            return data, label
        else:
            return data

    def make_new_labels_to_curr_obj(self, original_labels: torch.Tensor, all_object_list: list):
        """take original label from the data set, assign new labels by all objects in old+new order"""
        if original_labels is None:
            return original_labels
        obj_flattened = [SORTED_DATA_OBJ_LIST[item] for item in original_labels.flatten()]
        relative_labels = np.array([all_object_list.index(obj) for obj in obj_flattened])
        relative_labels = torch.tensor(relative_labels, dtype=original_labels.dtype, device=original_labels.device)
        logging.debug(f"relative_labels: \n    {relative_labels}")
        return relative_labels.reshape(original_labels.shape)

    def get_data_and_convert_labels(self, behavior_list=configs.behavior_list,
                                    source_tool_list=configs.source_tool_list,
                                    target_tool_list=configs.target_tool_list, modality_list=configs.modality_list,
                                    old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                                    trail_list=configs.trail_list, test_target=False):
        """
        :return: data[behavior_index][tool_index][object_index][trail_index]
         source_data: data from old_object_list + new_object_list
         truth_source: labels for old_object_list + new_object_list,
                        index starts from 0 to len(old_object_list + new_object_list) - 1
         target_data: if test_target==True, data from new_object_list, else from old_object_list
         truth_target: if test_target==True, labels for new_object_list,
                        index starts from len(old_object_list) to len(old_object_list + new_object_list)-1;
                       else for old_object_list, index starts from 0 to len(old_object_list) - 1
        """
        logging.debug(f"➡get_data_and_convert_labels...")
        assert behavior_list and trail_list and modality_list
        assert len(behavior_list) == 1  # for now, one behavior only
        assert len(target_tool_list) in [0, 1]  # at most one target tool at a time
        if test_target:
            assert len(new_object_list) != 0

        logging.debug(f"get source data: {source_tool_list}: {old_object_list + new_object_list}")
        source_data, truth_source = self.get_data(
            behavior_list, source_tool_list, modality_list,
            old_object_list + new_object_list, trail_list, get_labels=True)
        if source_data is not None:
            logging.debug(f"structured source tool data shape: {source_data.shape}")
            logging.debug(f"structured source tool label shape: {truth_source.shape}")
        else:
            logging.debug("no source data.")

        target_obj_list = new_object_list if test_target else old_object_list
        logging.debug(f"get target data: {target_tool_list}: {target_obj_list}")
        target_data, truth_target = self.get_data(
            behavior_list, target_tool_list, modality_list, target_obj_list, trail_list, get_labels=True)
        if target_data is not None:
            logging.debug(f"structured target tool data shape: {target_data.shape}")
            logging.debug(f"structured target tool label shape: {truth_target.shape}")
        else:
            logging.debug("no target data.")

        # convert label from original to the order of object list, always in the order of old+new
        all_obj_list = old_object_list + new_object_list
        truth_source = self.make_new_labels_to_curr_obj(original_labels=truth_source, all_object_list=all_obj_list)
        truth_target = self.make_new_labels_to_curr_obj(original_labels=truth_target, all_object_list=all_obj_list)

        return source_data, target_data, truth_source, truth_target

    def get_embeddings_and_labels(self, Encoder, source_data, target_data, truth_source, truth_target, l2_norm=False):
        # flatten structured data by trial(sample): (num_trials, emb_dim)
        if source_data is not None:
            encoded_source = Encoder(source_data).reshape(-1, configs.encoder_output_dim)
            truth_source = truth_source.reshape(-1, 1)
        else:
            encoded_source = torch.empty((0, configs.encoder_output_dim)).to(configs.device)
            truth_source = torch.empty((0, 1), dtype=torch.int64).to(configs.device)

        if target_data is not None:
            encoded_target = Encoder(target_data).reshape(-1, configs.encoder_output_dim)
            truth_target = truth_target.reshape(-1, 1)
        else:
            encoded_target = torch.empty((0, configs.encoder_output_dim)).to(configs.device)
            truth_target = torch.empty((0, 1), dtype=torch.int64).to(configs.device)
        # concat source and target
        all_labels = torch.cat([truth_source, truth_target], dim=0)
        all_embeds = torch.cat([encoded_source, encoded_target], dim=0)
        if l2_norm:
            all_embeds = torch.nn.functional.normalize(all_embeds, p=2, dim=1)  # L2 norm

        return all_embeds, all_labels

    def encode_all_data(self, Encoder, new_obj_only=False, train_obj_only=False,
                        behavior_list=configs.behavior_list, source_tool_list=configs.source_tool_list,
                        target_tool_list=configs.target_tool_list, modality_list=configs.modality_list,
                        old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                        trail_list=configs.trail_list):
        logging.debug(f"➡️encode all data...")
        if new_obj_only:
            target_tool_train = []
            old_object_list = []
        else:
            target_tool_train = target_tool_list
        source_data, target_data, truth_source, truth_target = self.get_data_and_convert_labels(
            behavior_list=behavior_list,
            source_tool_list=source_tool_list,
            target_tool_list=target_tool_train,
            modality_list=modality_list,
            old_object_list=old_object_list,
            new_object_list=new_object_list,
            trail_list=trail_list)

        all_embeds, all_labels = self.get_embeddings_and_labels(Encoder, source_data, target_data,
                                                                truth_source, truth_target, l2_norm=False)
        all_embeds, all_labels = all_embeds.cpu().detach().numpy(), all_labels.cpu().detach().numpy()
        logging.debug(f"all_embeds vectors' shape: {all_embeds.shape}")
        if target_tool_list and not train_obj_only:
            source_data_test, target_data_test, truth_source_test, truth_target_test = self.get_data_and_convert_labels(
                behavior_list=behavior_list, source_tool_list=[], target_tool_list=target_tool_list,
                modality_list=modality_list, old_object_list=old_object_list,
                new_object_list=new_object_list, trail_list=trail_list, test_target=True)
            test_embeds, test_labels = self.get_embeddings_and_labels(Encoder, source_data=source_data_test,
                                                                      truth_source=truth_source_test,
                                                                      target_data=target_data_test,
                                                                      truth_target=truth_target_test, l2_norm=False)
            test_embeds, test_labels = test_embeds.cpu().detach().numpy(), test_labels.cpu().detach().numpy()
            logging.debug(f"test_embeds.shape: {test_embeds.shape}")
            # Combine embeddings and labels
            all_embeds = np.concatenate([all_embeds, test_embeds], axis=0)
            all_labels = np.concatenate([all_labels, test_labels], axis=0)
        else:
            truth_target_test = None

        logging.debug(f"all_labels in encode_all_data: \n      {np.squeeze(all_labels)}")
        source_len = 0 if truth_source is None else truth_source.numel()
        target_train_len = 0 if truth_target is None else truth_target.numel()
        target_test_len = 0 if truth_target_test is None else truth_target_test.numel()

        return all_embeds, all_labels, source_len, target_train_len, target_test_len

    # %%
