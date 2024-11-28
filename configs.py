import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # apple M1 chip
else:
    device = 'cpu'

#########################encoder parameters


encoder_hidden_dim = 1024
encoder_output_dim = 256

epoch_encoder = 1000
lr_encoder = 1e-3
############## TL loss parameters ##########
TL_margin = 1
pairs_per_batch_per_object = 10

############## SINCERE loss parameters ##########
sincere_temp = 0.1

#############classifier parameters #######
epoch_classifier = 1000
lr_classifier = 1e-1
val_portion = 0


#################################################
train_category_num = 9
val_category_num = 3
test_category_num = 3

new_object_num = 3


#########################################



