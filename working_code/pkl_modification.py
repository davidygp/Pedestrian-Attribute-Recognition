#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:43:33 2020

@author: jiahao
"""


import os
import pickle
import random
import shutil

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

peta_col = peta_pkl.attr_name

#%% [pa100k] pa100k image name list in csv
df_pa100k = pd.read_csv('working_code/pa100k_machine_annotation.csv')

df_attr = df_pa100k[peta_col]
# pa_col = list(df_attr.columns)
df_pa100k['image_id_renamed'] = 'pa' + df_pa100k['image_id'].astype(str)


#%% [pa100k] rename images and save in the new folder

# img_dir_origin = "data/PA100k/images"
# img_dir_renamed = "data/PA100k/images_renamed"

# if not os.path.exists(img_dir_renamed):
#     os.makedirs(img_dir_renamed)

# for filename in os.listdir(img_dir_origin):
#     if filename in list(df_pa100k['image_id']):
#         shutil.copy( os.path.join(img_dir_origin, filename), os.path.join(img_dir_renamed, 'pa'+filename) )
#         print(f"convert {filename} to 'pa'{filename}")


#%% [pa100k] append <i> image name, <ii> labels, <iii> partition

# <i> image name
peta_pkl.image_name.extend(list(df_pa100k['image_id_renamed']))
peta_pkl.image_name = peta_pkl.image_name

# <ii> labels
peta_pkl.label = np.append(peta_pkl.label, np.array(df_attr), axis=0)

# <iii> partition
list_pa100k = [x for x in range(19000,119000)]
temp_peta = peta_pkl.partition.trainval
peta_pkl.partition.trainval = []

for idx in range(5):
    random.shuffle(list_pa100k)
    temp_list = np.append(temp_peta[idx], np.array(list_pa100k))
    peta_pkl.partition.trainval.append(temp_list)

peta_pkl.root = './data/pa100k/images'

#%% [pa100k] save pkl
with open('./data/PETA/dataset_pa100k.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)

#%% [RAP] (with pa100k) append <i> image name, <ii> labels, <iii> partition
with open('./data/PETA/dataset_pa100k.pkl', 'rb') as f:
    peta_pkl = pickle.load(f)

df_rap = pd.read_csv('RAP_rare_labelled.csv')
df_attr = df_rap[peta_col]

# <i> image name
peta_pkl.image_name.extend(list(df_rap['Image']))
peta_pkl.image_name = peta_pkl.image_name

# <ii> labels
peta_pkl.label = np.append(peta_pkl.label, np.array(df_attr), axis=0)

# <iii> partition
list_rap = [x for x in range(119000,len(peta_pkl.label))]
temp_peta = peta_pkl.partition.trainval
peta_pkl.partition.trainval = []

for idx in range(5):
    random.shuffle(list_rap)
    temp_list = np.append(temp_peta[idx], np.array(list_rap))
    peta_pkl.partition.trainval.append(temp_list)

peta_pkl.root = './data/rap/images'


#%% [RAP] (with pa100k) save pkl and check
with open('./data/PETA/dataset_pa100k_rap.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)
    
#%% [RAP] append <i> image name, <ii> labels, <iii> partition
with open('./data/PETA/dataset_116.pkl', 'rb') as f:
    peta_pkl = pickle.load(f)
    
df_rap = pd.read_csv('RAP_rare_labelled.csv')
df_attr = df_rap[peta_col]

# <i> image name
peta_pkl.image_name.extend(list(df_rap['Image']))
peta_pkl.image_name = peta_pkl.image_name

# <ii> labels
peta_pkl.label = np.append(peta_pkl.label, np.array(df_attr), axis=0)

# <iii> partition
list_rap = [x for x in range(19000,len(peta_pkl.label))]
temp_peta = peta_pkl.partition.trainval
peta_pkl.partition.trainval = []

for idx in range(5):
    random.shuffle(list_rap)
    temp_list = np.append(temp_peta[idx], np.array(list_rap))
    peta_pkl.partition.trainval.append(temp_list)

peta_pkl.root = './data/rap/images'


#%% [RAP] save pkl and check
with open('./data/PETA/dataset_rap.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)
