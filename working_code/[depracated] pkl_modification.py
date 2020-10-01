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
with open('./data/PETA/dataset_116.pkl', 'rb') as f:
    peta_pkl = pickle.load(f)

original_test_partition = peta_pkl.partition.test[0]
original_train_partition = peta_pkl.partition.train[0]
original_val_partition = peta_pkl.partition.val[0]
original_val_partition = peta_pkl.partition.trainval[0]
#%% save png file name list

image_name_list_peta = sorted(os.listdir('./data/PETA/images'))
image_name_list_rap = sorted([e for e in os.listdir('./data/RAP/images') if e not in image_name_list_peta])
image_name_list = image_name_list_peta + image_name_list_rap

annotated_peta_df = pd.read_csv('./working_code/RAP.csv')
# peta_col_name = list(annotated_peta_df.columns)
annotated_peta_df.drop(columns=['image_id'], inplace=True)
annotated_peta_array = annotated_peta_df.to_numpy()

#%% append data to pkl


# for new_image_name in image_name_list:
#     peta_pkl.image_name.append(new_image_name)

# check
# len(peta_pkl.image_name)
# peta_pkl.image_name = peta_pkl.image_name

peta_pkl.image_name = []
peta_pkl.image_name = image_name_list
peta_pkl.label = np.append(peta_pkl.label, annotated_peta_array, axis=0)

list_rap = [x for x in range(0, len(peta_pkl.image_name))]
list_rap_train = [e for e in list_rap if e not in peta_pkl.partition.test[0]]

peta_pkl.partition.trainval = []

for idx in range(5):
    random.shuffle(list_rap_train)
    peta_pkl.partition.trainval.append(np.array(list_rap_train))

peta_pkl.root = './data/RAP/images'

#%% save pkl file
with open('./working_code/rap_added_peta.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)
    
#%% check pkl file
with open('./working_code/rap_added_peta.pkl', 'rb') as f:
    rap_pkl = pickle.load(f)