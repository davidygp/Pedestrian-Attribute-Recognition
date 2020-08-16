#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:40:11 2020

@author: jiahao
"""


import os
import pickle
import random

import numpy as np
import pandas as pd
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

#%% read pkl file
with open('PETA.pkl', 'rb') as f:
    peta_pkl = pickle.load(f)

#%% save png file name list
image_name_list = sorted(os.listdir('data/RAP/images'))
annotated_peta_df = pd.read_csv('data/RAP/RAP.csv')
peta_col_name = list(annotated_peta_df.columns)
annotated_peta_df.drop(columns=['image_id'], inplace=True)
annotated_peta_array = annotated_peta_df.to_numpy()

#%% append data to pkl
for new_image_name in image_name_list:
    peta_pkl.image_name.append(new_image_name)

# check
# len(peta_pkl.image_name)

peta_pkl.image_name = peta_pkl.image_name
peta_pkl.label = np.append(peta_pkl.label, annotated_peta_array, axis=0)

list_rap = [x for x in range(0, len(peta_pkl.image_name))]
peta_pkl.partition.train = []
peta_pkl.partition.val = []
peta_pkl.partition.trainval = []

for idx in range(5):
    random.shuffle(list_rap)

    train = np.array(list_rap[:18000])
    val = np.array(list_rap[18000:])

    trainval = np.array(list_rap[:18000] + list_rap[18000:])
    
    peta_pkl.partition.train.append(train)
    peta_pkl.partition.val.append(val)
    peta_pkl.partition.trainval.append(trainval)

peta_pkl.save_dir = './data/RAP'
peta_pkl.root = './data/RAP/images'

#%%
with open('rap_added_peta.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)


