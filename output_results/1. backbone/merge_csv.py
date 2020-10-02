#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:19:15 2020

@author: jiahao
"""


import os
import glob
import pandas as pd

#%% merge csv with file name as a new column
os.chdir("./csv_folder")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

data = [] # pd.concat takes a list of dataframes as an agrument
for csv in all_filenames:
    frame = pd.read_csv(csv)
    frame['filename'] = os.path.basename(csv)
    data.append(frame)

df = pd.concat(data, ignore_index=True) #dont want pandas to try an align row indexes
df.to_csv("merged_data.csv", index=False, encoding='utf-8-sig')

#%% 
idx = df.groupby(['filename'])['valid_instance_f1'].transform(max) == df['valid_instance_f1']
result = df[idx]

result.to_csv("max_f1.csv", index=False)
