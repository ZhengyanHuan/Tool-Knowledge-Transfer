#%%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import configs
import model
from sincere_loss_class import SINCERELoss

#%%
class Tool_Knowledge_transfer_class():
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

        print('behaviors: ', len(self.behaviors), self.behaviors)
        print('objects: ', len(self.objects), self.objects)
        print('tools: ', len(self.tools), self.tools)
        print('trials: ', len(self.trials), self.trials)

        ####
        self.CEloss = torch.nn.CrossEntropyLoss()
        # self.Classifier = model.classifier(configs.encoder_output_dim, configs.new_object_num)

    def plot_func(self, record, type,save_name = 'test', plot_every = 10): #type-> 'encoder', 'classifier'
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 24

        color_group = ['red','blue']
        if type == 'encoder':
            xaxis = np.arange(1, len(record) +1)
            plt.plot(xaxis[::plot_every], record[::plot_every], color_group[0])
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title(f'Encoder Training Loss Progression \n '
                      f'Epochs: {configs.epoch_encoder}, lr: {configs.lr_encoder}')
            plt.grid()
            plt.savefig(r'./figs/'+save_name+'.png',  bbox_inches = 'tight')
            plt.show()
        elif type == 'classifier':
            xaxis = np.arange(1, record.shape[1] + 1)
            plt.plot(xaxis[::plot_every], record[0,::plot_every], color_group[0], label = 'tr loss')
            if record[1,-1] != 0:
                plt.plot(xaxis[::plot_every], record[1, ::plot_every], color_group[1], label = 'val loss')
            plt.xlabel('epochs')
            plt.title(f'Classifier Training Loss Progression \n '
                      f'Epochs: {configs.epoch_classifier}, lr: {configs.lr_classifier}')
            plt.ylabel('loss')
            plt.grid()
            plt.legend()
            plt.savefig(r'./figs/' + save_name + '.png', bbox_inches='tight')
            plt.show()
        else:
            print('invalid type')

    def prepare_data_classifier(self, behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder):
        train_test_index = int(len(trail_list) * (1 - configs.val_portion))

        source_data = self.get_data(behavior_list, source_tool_list, modality_list, new_object_list, trail_list)
        with torch.no_grad():
            encoded_source = Encoder(source_data)

        train_encoded_source = encoded_source[:, :, :, :train_test_index, :]
        val_encoded_source = encoded_source[:, :, :, train_test_index:, :]

        truth = np.zeros_like(encoded_source[:,:,:,:,-1].cpu())
        # truth = np.zeros([len(new_object_list), len(trail_list)])
        for i in range(len(new_object_list)):
            truth[:,:, i, :] = i
        truth = torch.tensor(truth, dtype=torch.int64, device=configs.device)

        train_truth = truth[:, :train_test_index]
        val_truth = truth[:, train_test_index:]

        return train_encoded_source, val_encoded_source, train_truth.reshape(-1), val_truth.reshape(-1)

    def train_classifier(self,behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder,lr_clf = configs.lr_classifier):
        loss_record = np.zeros([2, configs.epoch_classifier])


        train_encoded_source, val_encoded_source, train_truth_flat, val_truth_flat = self.prepare_data_classifier(behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder)
        Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)

        optimizer = optim.AdamW(Classifier.parameters(), lr=lr_clf)

        for i in range(configs.epoch_classifier):
            pred_tr = Classifier(train_encoded_source)
            pred_flat_tr = pred_tr.view(-1, len(new_object_list))
            loss_tr = self.CEloss(pred_flat_tr, train_truth_flat)
            loss_record[0,i] = loss_tr.detach().cpu().numpy()

            if len(val_truth_flat>0):
                with torch.no_grad():
                    pred_val = Classifier(val_encoded_source)
                    pred_flat_val = pred_val.view(-1, len(new_object_list))
                    loss_val = self.CEloss(pred_flat_val, val_truth_flat)
                    loss_record[1, i] = loss_val.detach().cpu().numpy()

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

            if (i+1)%1000 == 0:
                pred_label = torch.argmax(pred_flat_tr, dim=-1)
                correct_num = torch.sum(pred_label == train_truth_flat)
                accuracy_train = correct_num / len(train_truth_flat)

                print(f"epoch {i + 1}/{configs.epoch_classifier}, train loss: {loss_tr.item():.4f}, train accuracy: {accuracy_train.item() * 100 :.2f}%")
                if len(val_truth_flat>0):
                    pred_label = torch.argmax(pred_flat_val, dim=-1)
                    correct_num = torch.sum(pred_label == val_truth_flat)
                    accuracy_val= correct_num / len(val_truth_flat)

                    print(f"epoch {i + 1}/{configs.epoch_classifier}, val loss: {loss_val.item():.4f}, val accuracy: {accuracy_val.item() * 100 :.2f}%")

        self.plot_func(loss_record, 'classifier', f'classifier_{self.encoder_loss_fuc}')
        return Classifier

    def eval(self, Encoder, Classifier, behavior_list, target_tool_list,new_object_list, modality_list, trail_list):
        source_data = self.get_data(behavior_list, target_tool_list, modality_list, new_object_list, trail_list)
        truth_flat = np.zeros(len(trail_list)*len(new_object_list))
        for i in range(len(new_object_list)):
            truth_flat[i*len(trail_list):(i+1)*len(trail_list)] = i
        truth_flat = torch.tensor(truth_flat, dtype=torch.int64, device=configs.device)

        with torch.no_grad():
            encoded_source = Encoder(source_data)
            pred = Classifier(encoded_source)
        pred_flat = pred.view(-1, len(new_object_list))
        pred_label = torch.argmax(pred_flat, dim=-1)
        # torch.mps.synchronize()
        print(f"{len(truth_flat)} true labels     : {truth_flat.tolist()}")
        print(f"{len(pred_label)} predicted labels: {pred_label.tolist()}")
        correct_num = torch.sum(pred_label == truth_flat)
        accuracy_test = correct_num / len(truth_flat)
        print(f"test accuracy: {accuracy_test.item() * 100:.2f}%")
        return accuracy_test





    def train_encoder(self,behavior_list, source_tool_list, target_tool_list,old_object_list, modality_list, trail_list, lr_en = configs.lr_encoder):
        '''

        :param behavior_list:
        :param source_tool_list:
        :param target_tool_list:
        :param modality_list:
        :param old_object_list: e.g. ['chickpea', 'split-green-pea', 'glass-bead', 'chia-seed', 'wheat', 'wooden-button', 'styrofoam-bead', 'metal-nut-bolt', 'salt']
        :param trail_list: the index of training trails, e.g. [0,1,2,3,4,5,6,7]
        :return:
        '''
        loss_record = np.zeros(configs.epoch_encoder)
        new_object_list = []
        for object in self.objects:
            if object not in old_object_list:
                new_object_list.append(object)

        source_data = self.get_data(behavior_list, source_tool_list, modality_list, old_object_list + new_object_list , trail_list)
        target_data = self.get_data(behavior_list, target_tool_list, modality_list, old_object_list , trail_list)

        total_num_obj = len(old_object_list + new_object_list)
        truth_target = np.zeros(len(trail_list) * total_num_obj)
        for i in range(total_num_obj):
            truth_target[i * len(trail_list):(i + 1) * len(trail_list)] = i
        truth_target = torch.tensor(truth_target, dtype=torch.int64, device=configs.device)

        truth_source = np.zeros(len(trail_list) * len(old_object_list))
        for i in range(len(old_object_list)):
            truth_source[i * len(trail_list):(i + 1) * len(trail_list)] = i
        truth_source = torch.tensor(truth_source, dtype=torch.int64, device=configs.device)
        '''
        If we have more than one modality, we may need preprocessing and the input dim may not the 
        sum of data dim across all considered modalities. But I just put it here because we have 
        not figured out what to do.
        '''
        self.input_dim = 0
        for modality in modality_list:
            self.input_dim+=self.data_dict[behavior_list[0]][target_tool_list[0]][modality][old_object_list[0]]['X'][0].__len__()

        Encoder = model.encoder(self.input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
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
            if (i+1)%1000 == 0:
                print(f"epoch {i + 1}/{configs.epoch_encoder}, loss: {loss.item():.4f}")

        self.plot_func(loss_record, 'encoder', f'encoder_{self.encoder_loss_fuc}')
        return Encoder

    def sincere_ls_fn(self, source_data, truth_source, target_data, truth_target, Encoder, temperature=0.07):
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)
        tot_object_num = encoded_source.shape[2]
        old_object_num = encoded_target.shape[2]
        trail_num_per_object = encoded_source.shape[3]
        encoded_source = encoded_source.reshape(tot_object_num*trail_num_per_object, -1)
        encoded_target = encoded_target.reshape(old_object_num * trail_num_per_object, -1)

        all_embeds = torch.cat([encoded_source, encoded_target], dim=0)
        all_embeds_norm = torch.nn.functional.normalize(all_embeds, p=2, dim=1)  # L2 norm
        all_labels = torch.cat([truth_source, truth_target], dim=0)

        sincere_loss = SINCERELoss(temperature)
        return sincere_loss(all_embeds_norm, all_labels)

    def  get_same_object_list(self, encoded_source, encoded_target):
        same_object_list = []
        tot_len = encoded_source.shape[2]
        target_len = encoded_target.shape[2]
        for i in range(tot_len):
            if i < target_len:
                object_list1 = encoded_source[:,:,i,:,:].reshape([-1,configs.encoder_output_dim])
                object_list2 = encoded_target[:,:,i,:,:].reshape([-1,configs.encoder_output_dim])
                object_list = torch.concat([object_list1, object_list2], dim=0)
            else:
                object_list = encoded_source[:,:,i,:,:].reshape([-1,configs.encoder_output_dim])
            same_object_list.append(object_list)


        return same_object_list

    def TL_loss_fn(self, source_data, target_data, Encoder, alpha = configs.TL_margin):
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)
        same_object_list = self.get_same_object_list(encoded_source, encoded_target)

        trail_tot_num_list = np.array([same_object_list[i].shape[0] for i in range(same_object_list.__len__())])
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
            A_mat[object_index * configs.pairs_per_batch_per_object: (object_index + 1) * configs.pairs_per_batch_per_object] = object_list[A_index,:]
            P_mat[object_index * configs.pairs_per_batch_per_object: (object_index + 1) * configs.pairs_per_batch_per_object] = object_list[P_index,:]

            # Sample negative
            N_object_list = np.random.choice(trail_tot_num_list.__len__(), size=configs.pairs_per_batch_per_object)
            N_list = torch.zeros(configs.pairs_per_batch_per_object, configs.encoder_output_dim, dtype=torch.float32).to(configs.device)
            for i in range(len(N_object_list)):
                N_object_index = N_object_list[i]
                N_trail_index = np.random.choice(trail_tot_num_list[N_object_index])
                N_list[i] = same_object_list[N_object_index][N_trail_index]
                N_mat[object_index * configs.pairs_per_batch_per_object: (object_index + 1) * configs.pairs_per_batch_per_object] = N_list

        dPA = torch.norm(A_mat- P_mat, dim=1)
        dNA = torch.norm(A_mat- N_mat, dim=1)

        d = dPA - dNA + alpha
        d[d<0] = 0

        loss = torch.mean(d)
        return loss

    def get_data(self,behavior_list, tool_list, modality_list, object_list, trail_list):


        if len(modality_list) == 1:
            data_dim = self.data_dict[behavior_list[0]][tool_list[0]][modality_list[0]][object_list[0]]['X'][0].__len__()
            data = np.zeros((len(behavior_list), len(tool_list), len(object_list), len(trail_list), data_dim))
            '''
            Now we have 1 behavior, 1 tool. The data dim is 1x1xtrail_num x data_dim
            But this can work for multiple behaviors, tools
            '''
            for behavior_index in range(len(behavior_list)):
                behavior = behavior_list[behavior_index]
                for tool_index in range(len(tool_list)):
                    tool = tool_list[tool_index]
                    for object_index in range(len(object_list)):
                        object = object_list[object_index]
                        for trail_index in range(len(trail_list)):
                            trail = trail_list[trail_index]
                            data[behavior_index][tool_index][object_index][trail_index] = self.data_dict[behavior][tool][modality_list[0]][object]['X'][trail]

            data = torch.tensor(data, dtype=torch.float32, device=configs.device)

        else:
            data = None
            '''
            if we have more than one modality, the data dim are different and a tensor cannot hold this.
            So I leave this for future extension.
            '''

        return data








#%%




