#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:37:27 2020

@author: jiahao
"""


import os
import pickle
import random

import numpy as np
import pandas as pd
from easydict import EasyDict
from scipy.io import loadmat

#%%
with open('/Users/jiahao/Documents/GitHub/Pedestrian-Attribute-Recognition/data/PETA/dataset_116.pkl', 'rb') as f:
    peta_116 = pickle.load(f)

with open('/Users/jiahao/Documents/GitHub/Pedestrian-Attribute-Recognition/data/PETA/dataset_116_old.pkl', 'rb') as f:
    peta_116_old = pickle.load(f)

with open('/Users/jiahao/Documents/GitHub/Pedestrian-Attribute-Recognition/data/PETA/dataset_original.pkl', 'rb') as f:
    peta_original = pickle.load(f)    

#%%
"true" if peta_original.attr_name == peta_116_old.attr_name else "false"
"true" if peta_116.attr_name == peta_116_old.attr_name else "false"


"true" if np.allclose(peta_116.partition.trainval, peta_original.partition.trainval) else "false"
"true" if np.allclose(peta_116.partition.test, peta_original.partition.test) else "false"


"true" if np.allclose(peta_116.partition.trainval, peta_116_old.partition.trainval) else "false"
"true" if np.allclose(peta_116.partition.test, peta_116_old.partition.test) else "false"


"true" if np.allclose(peta_116.label, peta_original.label) else "false"

#%%

with open('/Users/jiahao/Documents/GitHub/Pedestrian-Attribute-Recognition/data/PETA/dataset_pa100k_rap.pkl', 'rb') as f:
    test1 = pickle.load(f)

with open('/Users/jiahao/Documents/GitHub/Pedestrian-Attribute-Recognition/data/PETA/dataset_pa100k_rap.pkl 2', 'rb') as f:
    test2 = pickle.load(f)