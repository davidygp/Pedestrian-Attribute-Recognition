#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:23:10 2020

@author: jiahao
"""


import pandas as pd

#%% Jiahao - RAP
df = pd.read_csv('working_code/RAPv2.csv')

attr_list = ['hs-Mask',
             'hs-Muffler',
             'shoes-ColorBlue',
            'shoes-ColorOrange',
            'shoes-ColorPurple',
            'lb-ColorGreen',
            'lb-ColorOrange',
            'lb-ColorRed',
            'lb-ColorYellow',
            'AgeLess16']

for attr in attr_list:
    x = df[df[attr]==1].count()[attr]
    print(attr, ": ", x)

#%% Jiahao - PA100K
df = pd.read_csv('working_code/PA100K_train.csv')

attr_list = ['LowerStripe']

for attr in attr_list:
    x = df[df[attr]==1].count()[attr]
    print(attr, ": ", x)
    

#%% copy jpg out
import shutil
import os

files = ['xxx.jpg', 'yyy.jpy']
dest = '/Users/jiahao/folder'

for file in files:
    shutil.copy(file, dest)