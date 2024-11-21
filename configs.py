import numpy as np
#########################encoder/classifier parameters

device = 'cuda:0'
encoder_hidden_dim = 16
encoder_output_dim = 4

train_category_num = 9
val_category_num = 3
test_category_num = 3

new_object_num = 3

epoch_encoder = 10000
lr_encoder = 1e-4
epoch_classifier = 10000
lr_classifier = 1e-1
#################################################

############## TL loss parameters ##########
TL_margin = 1
pairs_per_batch_per_object = 10

#########################################