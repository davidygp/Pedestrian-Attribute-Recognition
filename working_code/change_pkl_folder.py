#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:45:27 2020

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

with open('./data/PETA+RAP/dataset.pkl', 'rb') as f:
    peta_pkl = pickle.load(f)
    
peta_pkl.root = './data/PETA+RAP/images'

with open('./data/PETA+RAP/dataset_new.pkl', 'wb') as f:
    pickle.dump(peta_pkl, f)