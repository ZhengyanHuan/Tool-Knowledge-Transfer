
import torch
import model
import configs
from transfer_class import Tool_Knowledge_transfer_class

behavior_list = ['1-look']
source_tool_list = ['plastic-spoon']
target_tool_list = ['metal-scissor']
modality_list = ['effort']
trail_list = [0,1,2,3,4,5,6,7,8,9]

old_object_list = ['chickpea', 'split-green-pea', 'glass-bead', 'chia-seed', 'wheat', 'wooden-button', 'styrofoam-bead', 'metal-nut-bolt', 'salt']
new_object_list = ['detergent', 'empty', 'plastic-bead']
myclass = Tool_Knowledge_transfer_class()

input_dim = 0
for modality in modality_list:
    input_dim+=myclass.data_dict['1-look']['metal-scissor'][modality]['metal-nut-bolt']['X'][0].__len__()

#%%
myencoder = myclass.train_encoder(behavior_list, source_tool_list, target_tool_list,old_object_list, modality_list, trail_list)
pt = "myencoder.pt"
torch.save(myencoder.state_dict(), './saved_model/encoder/'+ pt)




#%%
Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
Encoder.load_state_dict(torch.load('./saved_model/encoder/myencoder.pt'))
myclassifier = myclass.train_classifier(behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder)
pt = "myclassifier.pt"
torch.save(myclassifier.state_dict(), './saved_model/encoder/'+ pt)


#%%
Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
Encoder.load_state_dict(torch.load('./saved_model/encoder/myencoder.pt'))

Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)
Classifier.load_state_dict(torch.load('./saved_model/encoder/myclassifier.pt'))

myclass.eval(Encoder, Classifier, behavior_list, target_tool_list,new_object_list, modality_list, trail_list)
