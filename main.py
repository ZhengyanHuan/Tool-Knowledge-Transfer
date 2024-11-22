
import torch
import model
import configs
from transfer_class import Tool_Knowledge_transfer_class
import time


# start_time = time.time()

behavior_list = ['3-stirring-fast']
source_tool_list = ['plastic-spoon']
target_tool_list = ['metal-scissor']
modality_list = ['audio']
trail_list = [0,1,2,3,4,5,6,7,8,9]

old_object_list = ['chickpea', 'split-green-pea', 'glass-bead', 'chia-seed', 'wheat', 'wooden-button', 'styrofoam-bead', 'metal-nut-bolt', 'salt']
new_object_list = ['detergent', 'empty', 'plastic-bead']
myclass = Tool_Knowledge_transfer_class()

input_dim = 0
for modality in modality_list:
    input_dim+=myclass.data_dict['1-look']['metal-scissor'][modality]['metal-nut-bolt']['X'][0].__len__()

encoder_pt_name = "myencoder.pt"
clf_pt_name = "myclassifier.pt"
retrain = True

#%%
if retrain:
    print(f"training representation encoder...")
    encoder_time = time.time()
    myencoder = myclass.train_encoder(behavior_list, source_tool_list, target_tool_list,old_object_list, modality_list, trail_list)
    torch.save(myencoder.state_dict(), './saved_model/encoder/'+ encoder_pt_name)
    print(f"Time used for encoder training: {round((time.time() - encoder_time)//60)} min {(time.time() - encoder_time)%60:.1f} sec.")

#%%
if retrain:
    print(f"training classification head...")
    clf_time = time.time()

    Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
    Encoder.load_state_dict(torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))
    myclassifier = myclass.train_classifier(behavior_list, source_tool_list,new_object_list, modality_list, trail_list, Encoder)
    torch.save(myclassifier.state_dict(), './saved_model/classifier/' + clf_pt_name)

    print(f"Time used for classifier training: {round((time.time() - clf_time)//60)} min {(time.time() - clf_time)%60:.1f} sec.")

#%%
start_time = time.time()
Encoder = model.encoder(input_dim, configs.encoder_output_dim, configs.encoder_hidden_dim).to(configs.device)
Encoder.load_state_dict(torch.load('./saved_model/encoder/' + encoder_pt_name, map_location=torch.device(configs.device)))

Classifier = model.classifier(configs.encoder_output_dim, len(new_object_list)).to(configs.device)
Classifier.load_state_dict(torch.load('./saved_model/classifier/' + clf_pt_name, map_location=torch.device(configs.device)))

print(f"Evaluating the classifier...")
myclass.eval(Encoder, Classifier, behavior_list, target_tool_list,new_object_list, modality_list, trail_list)
print(f"total time used: {round((time.time() - start_time)//60)} min {(time.time() - start_time)%60:.1f} sec.")