# %%
import logging
import os
import pickle
from typing import Union, Tuple

import numpy as np
import torch
import torch.optim as optim

import configs
import model
from helpers.data_helpers import sanity_check_data_labels
from helpers.viz_helpers import plot_learning_progression
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
        assert encoder_loss_fuc in ['TL', 'sincere']

        robots_data_filepath = r'data' + os.sep + data_name
        bin_file = open(robots_data_filepath, 'rb')
        robot = pickle.load(bin_file)
        bin_file.close()

        self.data_dict = robot
        sanity_check_data_labels(self.data_dict)
        self.encoder_loss_fuc = encoder_loss_fuc
        logging.info(f"Encoder loss function: {encoder_loss_fuc}")
        self.enc_l2_norm = self._decide_l2_norm()

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
        self.input_dim = 0

    def _decide_l2_norm(self):
        """might change this rule later"""
        return self.encoder_loss_fuc == "sincere"

    def _assign_labels_to_data(self, structured_data: torch.Tensor, object_list: list) -> torch.Tensor:
        """
        structured_data shape: [n_behavior, n_tools, len(object_list), n_trials, data_sample_dim]
        return label shape: [n_behavior, n_tools, len(object_list), n_trials]"""
        truth = np.zeros_like(structured_data[:, :, :, :, -1].cpu())
        for i in range(len(object_list)):
            truth[:, :, i, :] = i
        return torch.tensor(truth, dtype=torch.int64, device=configs.device)

    def _prepare_data_classifier(self, Encoder, behavior_list, source_tool_list, new_object_list,
                                 modality_list, trail_list, val_portion) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return:
        data shape: [n_behavior, n_tools, num_objects, n_trials, emb_dim],
        label shape: [n_behavior*n_tools*num_objects&n_trials, ]

        """
        logging.debug(f"➡️ prepare_data_classifier..")
        train_test_index = int(len(trail_list) * (1 - val_portion))
        logging.debug(f"get source data for classifier from {new_object_list}")
        source_data = self.get_data(behavior_list, source_tool_list, modality_list, new_object_list, trail_list)
        with torch.no_grad():
            Encoder.l2_norm = self.enc_l2_norm
            encoded_source = Encoder(source_data)

        train_encoded_source = encoded_source[:, :, :, :train_test_index, :]
        val_encoded_source = encoded_source[:, :, :, train_test_index:, :]

        truth = self._assign_labels_to_data(encoded_source, new_object_list)
        train_truth = truth[:, :train_test_index]
        val_truth = truth[:, train_test_index:]
        logging.debug(f"train_truth: \n      {train_truth}")
        logging.debug(f"val_truth: \n      {val_truth}")

        return train_encoded_source, val_encoded_source, train_truth.reshape(-1), val_truth.reshape(-1)

    def train_classifier(self, Encoder, behavior_list=configs.behavior_list, trail_list=configs.trail_list,
                         new_object_list=configs.new_object_list, modality_list=configs.modality_list,
                         source_tool_list=configs.source_tool_list, lr_clf=configs.lr_classifier,
                         epoch_classifier=configs.epoch_classifier, encoder_output_dim=configs.encoder_output_dim,
                         val_portion=configs.val_portion, plot_learning=True):
        configs.set_torch_seed()
        loss_record = np.zeros([2, epoch_classifier])
        logging.debug(f"➡️ train_classifier..")

        train_encoded_source, val_encoded_source, train_truth_flat, val_truth_flat = self._prepare_data_classifier(
            behavior_list=behavior_list, source_tool_list=source_tool_list, new_object_list=new_object_list,
            modality_list=modality_list, trail_list=trail_list, Encoder=Encoder, val_portion=val_portion)
        Classifier = model.classifier(encoder_output_dim, len(new_object_list)).to(configs.device)

        optimizer = optim.AdamW(Classifier.parameters(), lr=lr_clf)

        for i in range(epoch_classifier):
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

                logging.info(f"epoch {i + 1}/{epoch_classifier}, train loss: {loss_tr.item():.4f}, "
                             f"train accuracy: {accuracy_train.item() * 100 :.2f}%, "
                             f"random guess accuracy: {100 / len(new_object_list):.2f}%")
                if len(val_truth_flat > 0):
                    pred_label = torch.argmax(pred_flat_val, dim=-1)
                    correct_num = torch.sum(pred_label == val_truth_flat)
                    accuracy_val = correct_num / len(val_truth_flat)

                    logging.info(
                        f"epoch {i + 1}/{epoch_classifier}, val loss: {loss_val.item():.4f}, "
                        f"val accuracy: {accuracy_val.item() * 100 :.2f}%, "
                        f"random guess accuracy: {100 / len(new_object_list):.2f}%")
        if plot_learning:
            plot_learning_progression(record=loss_record, type='classifier',
                                      lr_classifier=lr_clf, encoder_output_dim=encoder_output_dim,
                                      loss_func=self.encoder_loss_fuc, TL_margin=None, sincere_temp=None,
                                      lr_encoder=None, save_name=f'classifier_{self.encoder_loss_fuc}')
        return Classifier

    def eval(self, Encoder, Classifier, tool_list=configs.target_tool_list, behavior_list=configs.behavior_list,
             new_object_list=configs.new_object_list, modality_list=configs.modality_list,
             trail_list=configs.trail_list, return_pred=False) -> Union[Tuple[float, list, list], float]:
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
            Encoder.l2_norm = self.enc_l2_norm
            encoded_source = Encoder(source_data)
            pred = Classifier(encoded_source)
        pred_flat = pred.view(-1, len(new_object_list))
        pred_label = torch.argmax(pred_flat, dim=-1)

        correct_num = torch.sum(pred_label == truth_flat)
        accuracy_test = (correct_num / len(truth_flat)).item()
        truth_flat = truth_flat.tolist()
        pred_label = pred_label.tolist()

        logging.info(f"{len(truth_flat)} true labels: {truth_flat}")
        logging.info(f"{len(pred_label)} pred labels: {pred_label}")
        logging.info(
            f"test accuracy: {accuracy_test * 100:.2f}%, random guess accuracy: {100 / len(new_object_list):.2f}%")
        if return_pred:
            return accuracy_test, pred_flat, pred_label
        else:
            return accuracy_test

    def train_encoder(self, behavior_list=configs.behavior_list,
                      source_tool_list=configs.source_tool_list, target_tool_list=configs.target_tool_list,
                      old_object_list=configs.old_object_list, new_object_list=configs.new_object_list,
                      modality_list=configs.modality_list, trail_list=configs.trail_list, lr_en=configs.lr_encoder,
                      TL_margin=configs.TL_margin, encoder_output_dim=configs.encoder_output_dim,
                      sincere_tem=configs.sincere_temp, epoch_encoder=configs.epoch_encoder,
                      pairs_per_batch_per_object=configs.pairs_per_batch_per_object, plot_learning=True):
        """
        :param lr_en: encoder learning rate
        :param new_object_list: list of objects that only source tool has
        :param old_object_list: list of objects that both tools share
        :param source_tool_list: tool(s) with data from old_object_list + new_object_list
        :param target_tool_list: tool(s) with data from  old_object_list
        :param behavior_list:
        :param modality_list:
        :param trail_list: the index of training trails, e.g. [0,1,2,3,4,5,6,7]
        :return: trained encoder
        """
        logging.debug(f"➡️ train_encoder..")
        loss_record = np.zeros(epoch_encoder)

        source_data, target_data, truth_source, truth_target = self._get_encoder_data_and_labels(
            behavior_list=behavior_list, trail_list=trail_list, modality_list=modality_list,
            source_tool_list=source_tool_list, target_tool_list=target_tool_list,
            old_object_list=old_object_list, new_object_list=new_object_list)
        '''
        If we have more than one modality, we may need preprocessing and the input dim may not the 
        sum of data dim across all considered modalities. But I just put it here because we have 
        not figured out what to do.
        '''
        self.input_dim = 0
        for modality in modality_list:
            self.input_dim += len(
                self.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0])

        Encoder = model.encoder(self.input_dim, l2_norm=self.enc_l2_norm).to(configs.device)
        Encoder.l2_norm = self.enc_l2_norm
        optimizer = optim.AdamW(Encoder.parameters(), lr=lr_en)

        for i in range(epoch_encoder):
            if self.encoder_loss_fuc == "TL":
                loss = self.TL_loss_fn(source_data, target_data, Encoder, alpha=TL_margin,
                                       encoder_output_dim=encoder_output_dim,
                                       pairs_per_batch_per_object=pairs_per_batch_per_object)
            elif self.encoder_loss_fuc == "sincere":
                loss = self.sincere_loss_fn(source_data, truth_source, target_data, truth_target, Encoder,
                                            temperature=sincere_tem, encoder_output_dim=encoder_output_dim)
            else:
                raise Exception(f"{self.encoder_loss_fuc} not available.")
            loss_record[i] = loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 500 == 0:
                logging.info(f"epoch {i + 1}/{epoch_encoder}, loss: {loss.item():.4f}")
        if plot_learning:
            plot_learning_progression(record=loss_record, type='encoder',
                                      lr_classifier=None, encoder_output_dim=encoder_output_dim,
                                      loss_func=self.encoder_loss_fuc, TL_margin=TL_margin, sincere_temp=sincere_tem,
                                      lr_encoder=lr_en, save_name=f'encoder_{self.encoder_loss_fuc}')
        return Encoder

    def sincere_loss_fn(self, source_data, truth_source, target_data, truth_target,
                        Encoder, temperature, encoder_output_dim) -> torch.Tensor:
        all_embeds_norm, all_labels = self._make_encoder_projections(
            Encoder=Encoder, source_data=source_data, target_data=target_data, truth_source=truth_source,
            truth_target=truth_target, encoder_output_dim=encoder_output_dim)
        sincere_loss = SINCERELoss(temperature)
        if len(all_labels.shape) > 1:
            all_labels = torch.squeeze(all_labels)  # labels (torch.tensor): (B,)
        return sincere_loss(all_embeds_norm, all_labels)

    def _get_same_object_list(self, encoded_source, encoded_target, encoder_output_dim):
        """
        input data shape: [n_behavior, n_tools, num_objects, n_trials, data_sample_dim]
        assuming that encoded_source is constructed in the order of old_object_list + new_object_list,
        and encoded_target in the order of old_object_list, so their old object part match
        """
        same_object_list = []
        tot_len = encoded_source.shape[2]
        target_len = encoded_target.shape[2]
        for i in range(tot_len):
            if i < target_len:
                object_list1 = encoded_source[:, :, i, :, :].reshape([-1, encoder_output_dim])
                object_list2 = encoded_target[:, :, i, :, :].reshape([-1, encoder_output_dim])
                object_list = torch.concat([object_list1, object_list2], dim=0)
            else:
                object_list = encoded_source[:, :, i, :, :].reshape([-1, encoder_output_dim])
            same_object_list.append(object_list)

        return same_object_list

    def TL_loss_fn(self, source_data, target_data, Encoder, alpha, encoder_output_dim,
                   pairs_per_batch_per_object) -> torch.Tensor:
        Encoder.l2_norm = self.enc_l2_norm
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)
        same_object_list = self._get_same_object_list(encoded_source, encoded_target, encoder_output_dim)

        trail_tot_num_list = np.array([same_object_list[i].shape[0] for i in range(len(same_object_list))])
        tot_object_num = encoded_source.shape[2]

        A_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)
        P_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)
        N_mat = torch.zeros(pairs_per_batch_per_object * tot_object_num, encoder_output_dim, device=configs.device)

        for object_index in range(tot_object_num):
            object_list = same_object_list[object_index]

            # Sample anchor and positive
            A_index = np.random.choice(trail_tot_num_list[object_index], size=pairs_per_batch_per_object)
            P_index = np.random.choice(trail_tot_num_list[object_index], size=pairs_per_batch_per_object)
            start = object_index * pairs_per_batch_per_object
            end = (object_index + 1) * pairs_per_batch_per_object
            A_mat[start: end] = object_list[A_index, :]
            P_mat[start: end] = object_list[P_index, :]

            # Sample negative
            N_object_list = np.random.choice(len(trail_tot_num_list), size=pairs_per_batch_per_object)
            N_list = torch.zeros(pairs_per_batch_per_object, encoder_output_dim,
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
                 trail_list=configs.trail_list, get_labels=False) \
            -> Union[None, torch.Tensor, Tuple[None, None], Tuple[torch.Tensor, torch.Tensor]]:
        """
        :return:
        data shape: None, or tensor of shape [n_behavior, n_tools, num_objects, n_trials, data_sample_dim],
        label shape: if get_labels, None or tensor of shape [n_behavior, n_tools, num_objects, n_trials, 1]

        for each behavior&tool, data is ordered by object_list, then trail_list
        we ASSUME the dataset labels are created using sorted 15 object names, i.e., "cane-suger" is 0.
        Note that the returned labels do NOT always start from 0
            i.e., object_list is a subset of the one used to create the labels in the dataset
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
                        for trial_index in range(len(trail_list)):
                            try:
                                trial = trail_list[trial_index]
                                data[behavior_index][tool_index][object_index][trial_index] = \
                                    self.data_dict[behavior][tool][modality_list[0]][object]['X'][trial]
                                label[behavior_index][tool_index][object_index][trial_index] = \
                                    self.data_dict[behavior][tool][modality_list[0]][object]['Y'][trial]
                            except Exception as e:
                                print(f"something wrong here: behavior: {behavior}, tool: {tool}, "
                                      f"modality: {modality_list[0]}, object: {object}, trail index: {trial_index}")
                                raise e

            data = torch.tensor(data, dtype=torch.float32, device=configs.device)
            label = torch.tensor(label, dtype=torch.int64, device=configs.device)
            logging.debug(f"data mata: {meta_data}")
            logging.debug(f"structured source tool data shape: {data.shape}")
            logging.debug(f"structured source tool label shape: {label.shape}")

        else:
            data = None
            label = None
            logging.debug("No data.")
            '''
            if we have more than one modality, the data dim are different and a tensor cannot hold this.
            So I leave this for future extension.
            '''

        if get_labels:
            return data, label
        else:
            return data

    def _get_encoder_data_and_labels(self, behavior_list, source_tool_list, target_tool_list, modality_list,
                                     old_object_list, new_object_list,trail_list):
        """
        :return:
        data shape: [n_behavior, n_tools, num_objects, n_trials, data_sample_dim],
        label shape: [n_behavior, n_tools, num_objects, n_trials, 1], always starts from 0

         source_data: data from source tool & old_object_list + new_object_list
         truth_source: labels for source tool & old_object_list + new_object_list,
                        index starts from 0 to len(old_object_list + new_object_list) - 1
         target_data: data from target tool & new_object_list
         truth_target: labels for target tool & new_object_list, index starts from 0 to len(old_object_list) - 1
        """
        logging.debug(f"➡️ get_data_and_convert_labels...")
        assert behavior_list and trail_list and modality_list
        assert len(behavior_list) == 1  # for now, one behavior only
        all_obj_list = old_object_list + new_object_list  # has to be this order
        # =========  get source tool data
        logging.debug(f"get source data for encoder: {source_tool_list}: {all_obj_list}")
        source_data = self.get_data(behavior_list, source_tool_list, modality_list, all_obj_list, trail_list)
        # ========= get target tool data, get shared object by default
        logging.debug(f"get target data for encoder: {target_tool_list}: {old_object_list}")
        target_data = self.get_data(behavior_list, target_tool_list, modality_list, old_object_list, trail_list)

        truth_source, truth_target = None, None
        if source_data is not None:
            truth_source = self._assign_labels_to_data(structured_data=source_data, object_list=all_obj_list)
        if target_data is not None:
            truth_target = self._assign_labels_to_data(structured_data=target_data, object_list=old_object_list)

        return source_data, target_data, truth_source, truth_target

    def _make_encoder_projections(self, Encoder, source_data, target_data, truth_source, truth_target,
                                  encoder_output_dim, assist_data=None, truth_assist=None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        encode structured data [n_behavior, n_tools, num_objects, n_trials, data_sample_dim] ,
        and flatten the embeddings to [num_samples, emb_dim] , i.e., [n_behavior*n_tools*num_objects*n_trials, emb_dim]
        :return all_embeds: [num_samples, emb_dim], all_labels: [num_samples, 1]
        """
        Encoder.l2_norm = self.enc_l2_norm
        if source_data is not None:
            encoded_source = Encoder(source_data).reshape(-1, encoder_output_dim)
            truth_source = truth_source.reshape(-1, 1)
        else:
            encoded_source = torch.empty((0, encoder_output_dim)).to(configs.device)
            truth_source = torch.empty((0, 1), dtype=torch.int64).to(configs.device)

        if assist_data is not None:
            encoded_assist = Encoder(assist_data).reshape(-1, encoder_output_dim)
            truth_assist = truth_assist.reshape(-1, 1)
        else:
            encoded_assist = torch.empty((0, encoder_output_dim)).to(configs.device)
            truth_assist = torch.empty((0, 1), dtype=torch.int64).to(configs.device)

        if target_data is not None:
            encoded_target = Encoder(target_data).reshape(-1, encoder_output_dim)
            truth_target = truth_target.reshape(-1, 1)
        else:
            encoded_target = torch.empty((0, encoder_output_dim)).to(configs.device)
            truth_target = torch.empty((0, 1), dtype=torch.int64).to(configs.device)
        # concat source and target
        all_labels = torch.cat([truth_source, truth_assist, truth_target], dim=0)
        all_embeds = torch.cat([encoded_source, encoded_assist, encoded_target], dim=0)

        return all_embeds, all_labels

    # %%
