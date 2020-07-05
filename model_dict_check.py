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


# model = se_resnet50()
model = dpn68()
# model = resnet50()

#%%
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    
    
# def remove_fc(state_dict):
#     return {key: value for key, value in state_dict.items() if not key.startswith('classifier.')}

# model.load_state_dict(remove_fc(model.state_dict()))
