#%%
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import configs
import model

#%%
class Tool_Knowledge_transfer_class():
    def __init__(self):

        ####load dataset
        robots_data_filepath = r'data' + os.sep + 'dataset_discretized.bin'
        bin_file = open(robots_data_filepath, 'rb')
        robot = pickle.load(bin_file)
        bin_file.close()

        self.data_dict = robot

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


    def train_classifier(self,behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder):
        source_data = self.get_data(behavior_list, source_tool_list, modality_list, new_object_list, trail_list)
        with torch.no_grad():
            encoded_source = Encoder(source_data)

        Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)

        truth_flat = torch.zeros(len(trail_list)*len(new_object_list), dtype=torch.int64, device=configs.device)
        for i in range(len(new_object_list)):
            truth_flat[i*len(trail_list):(i+1)*len(trail_list)] = i
        # truth_flat = truth.view(-1)
        optimizer = optim.AdamW(Classifier.parameters(), lr=configs.lr_classifier)

        for i in range(configs.epoch_classifier):
            pred = Classifier(encoded_source)

            pred_flat = pred.view(-1, len(new_object_list))
            loss = self.CEloss(pred_flat, truth_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss)
            if (i+1)%1000 == 0:
                pred_label = torch.argmax(pred_flat, dim = -1)
                correct_num = torch.sum(pred_label == truth_flat)
                accuracy_train = correct_num/ len(truth_flat)
                print(f"epoch {i + 1}/{configs.epoch_classifier}, loss: {loss.item():.4f}, train accuracy: {accuracy_train.item() * 100 :.2f}%")
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





    def train_encoder(self,behavior_list, source_tool_list, target_tool_list,old_object_list, modality_list, trail_list):
        '''

        :param behavior_list:
        :param source_tool_list:
        :param target_tool_list:
        :param modality_list:
        :param old_object_list: e.g. ['chickpea', 'split-green-pea', 'glass-bead', 'chia-seed', 'wheat', 'wooden-button', 'styrofoam-bead', 'metal-nut-bolt', 'salt']
        :param trail_list: the index of training trails, e.g. [0,1,2,3,4,5,6,7]
        :return:
        '''

        new_object_list = []
        for object in self.objects:
            if object not in old_object_list:
                new_object_list.append(object)

        source_data = self.get_data(behavior_list, source_tool_list, modality_list, old_object_list + new_object_list , trail_list)
        target_data = self.get_data(behavior_list, target_tool_list, modality_list, old_object_list , trail_list)

        '''
        If we have more than one modality, we may need preprocessing and the input dim may not the 
        sum of data dim across all considered modalities. But I just put it here because we have 
        not figured out what to do.
        '''
        self.input_dim = 0
        for modality in modality_list:
            self.input_dim+=self.data_dict['1-look']['metal-scissor'][modality]['metal-nut-bolt']['X'][0].__len__()

        Encoder = model.encoder(self.input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
        optimizer = optim.AdamW(Encoder.parameters(), lr=configs.lr_encoder)

        for i in range(configs.epoch_encoder):
            loss = self.TL_loss_fn(source_data, target_data, Encoder)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                print(f"epoch {i + 1}/{configs.epoch_encoder}, loss: {loss.item():.4f}")

        return Encoder

    def TL_loss_fn(self, source_data, target_data, Encoder):
        encoded_source = Encoder(source_data)
        encoded_target = Encoder(target_data)

        tot_object_num = encoded_source.shape[2]
        old_object_num = encoded_target.shape[2]
        trail_num_per_object = encoded_source.shape[3]
        encoded_tot = torch.concat([encoded_source, encoded_target], dim = 2)
        '''
        concat at the dim of object, e.g. there is 9 old objs
        0-14 denote the 15 objects of the source, 15-23 denote the 9 objs from the target 
        Let n denote the obj index in encoded_tot, if n%15 is the same, the corresponded objs are the same
        '''

        A_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)
        P_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)
        N_mat = torch.zeros(configs.pairs_per_batch_per_object * tot_object_num, configs.encoder_output_dim,
                            device=configs.device)

        for object_index in range(tot_object_num):


            if object_index<old_object_num:
                PA_valid_index = np.array([object_index, object_index + tot_object_num])
            else:
                PA_valid_index = np.array([object_index])

            st_index = np.random.choice(PA_valid_index, size = configs.pairs_per_batch_per_object * 2)
            trial_index = np.random.randint(0, trail_num_per_object, size=configs.pairs_per_batch_per_object * 2)

            '''
            behavior index and tool index are set to default 0 for now
            I understand it is possible to have duplicated A and P, but with a not big probability. 
            '''
            A_mat[object_index * configs.pairs_per_batch_per_object: (object_index+1) * configs.pairs_per_batch_per_object] = encoded_tot[0,0,st_index[:configs.pairs_per_batch_per_object], trial_index[:configs.pairs_per_batch_per_object],:]
            P_mat[object_index * configs.pairs_per_batch_per_object: (object_index+1) * configs.pairs_per_batch_per_object] = encoded_tot[0,0,st_index[configs.pairs_per_batch_per_object:], trial_index[configs.pairs_per_batch_per_object:],:]

            all_index = np.arange(0, tot_object_num + old_object_num)
            N_valid_index = np.setdiff1d(all_index, PA_valid_index)

            st_index_N = np.random.choice(N_valid_index, size = configs.pairs_per_batch_per_object)
            trial_index_N = np.random.randint(0, trail_num_per_object, size=configs.pairs_per_batch_per_object)
            N_mat[object_index * configs.pairs_per_batch_per_object: (object_index + 1) * configs.pairs_per_batch_per_object] = encoded_tot[0,0,st_index_N,trial_index_N,:]

        dPA = torch.norm(A_mat- P_mat, dim=1)
        dNA = torch.norm(A_mat- N_mat, dim=1)

        d = dPA - dNA + configs.TL_margin
        d[d<0] = 0

        loss = torch.mean(d)
        return loss

    def get_data(self,behavior_list, tool_list, modality_list, object_list, trail_list):


        if len(modality_list) == 1:
            data_dim = self.data_dict['1-look']['metal-scissor'][modality_list[0]]['metal-nut-bolt']['X'][0].__len__()
            data = torch.zeros(len(behavior_list), len(tool_list), len(object_list), len(trail_list), data_dim, device=configs.device)
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
                            data[behavior_index][tool_index][object_index][trail_index] = torch.tensor(self.data_dict[behavior][tool][modality_list[0]][object]['X'][trail], dtype=torch.float32, device=configs.device)


        else:
            data = None
            '''
            if we have more than one modality, the data dim are different and a tensor cannot hold this.
            So I leave this for future extension.
            '''

        return data








#%%




