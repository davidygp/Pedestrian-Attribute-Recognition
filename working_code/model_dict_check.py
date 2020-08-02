#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:07:08 2020

@author: jiahao
"""
#%% 
from models.densenet import densenet121
from models.resnet import resnet50
# from models.senet import se_resnet101, se_resnet50
# from models.senet_origin import se_resnet101, se_resnet50
from models.resnext import resnext101_32x4d
from models.dpn import dpn68

# load the model

# model = se_resnet50()
model = dpn68()
# model = resnet50()

#%%

###
# build customized model here
###

import torchvision.models as models

pretrained_model = models.resnet50(pretrained=True)
print(pretrained_model) #print model structure

# load pretrained wieght
pretrained_dict = pretrained_model.state_dict()

# check model dict
for k, v in pretrained_dict.items():
    print(k)  #print name of each layer

# check model dict with size
for k in pretrained_dict:
    print(k, "\t", pretrained_dict[k].size())

##################
## remove model fc layer 
# def remove_fc(state_dict):
#     return {key: value for key, value in state_dict.items() if not key.startswith('classifier.')}

# model.load_state_dict(remove_fc(model.state_dict()))
##################
    
# check weight with the layer name listed above
pretrained_dict['fc.bias']

model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)

