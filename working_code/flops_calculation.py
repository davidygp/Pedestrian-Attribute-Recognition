#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:38:38 2020

@author: jiahao
"""
import os
import pandas as pd
import torch

from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from models.dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from models.densenet import densenet121, densenet169, densenet201, densenet161
from models.senet import se_resnet101, se_resnet50


import sys

import argparse

import torchvision.models as models

from ptflops import get_model_complexity_info

    
#%% our model flops

FORCE_TO_CPU = True
models = ['resnet101',
        'dpn107',
        'dpn131',
        'resnext101_32x8d',
        'se_resnet50',
        'se_resnet101',
        'resnet50',
        'dpn92',
        'resnext50_32x4d',
        'dpn68',
        'dpn68b',
        'densenet121']

parser = argument_parser()

df = pd.DataFrame()
data = []

for trained_model in models:
    save_model_path = os.path.join('backbone_model', trained_model+'.pth')
    args = parser.parse_args(['PETA', '--model='+trained_model])
    print(save_model_path)
    print(args)

    _, predict_tsfm = get_transform(args)
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=predict_tsfm)
    
    args.att_list = valid_set.attr_id
    
    backbone = getattr(sys.modules[__name__], args.model)()
    if "dpn68" in args.model:
        net_parameter = 832
    elif "dpn" in args.model:
        net_parameter = 2688
    elif "densenet" in args.model:
        net_parameter = 1024
    else:
        net_parameter = 2048
        
    classifier = BaseClassifier(netpara=net_parameter, nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    
    if torch.cuda.is_available() and not FORCE_TO_CPU:
        model = torch.nn.DataParallel(model).cuda()
        ckpt = torch.load(save_model_path)
        print(f'Model is served with GPU ')
    else:
        model = torch.nn.DataParallel(model)
        ckpt = torch.load(save_model_path, map_location=torch.device('cpu'))
        print(f'Model is served with CPU ')
    
    model.load_state_dict(ckpt['state_dicts'])
    model.eval()
    
    macs, params = get_model_complexity_info(model, (3, 256, 192),
                                             as_strings=True,
                                             print_per_layer_stat=False,
                                             verbose=False)
    data.append([trained_model, macs, params])

df = pd.DataFrame(data, columns=['model','macs','params'])
df.to_csv('flops.csv')