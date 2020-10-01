#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:19:15 2020

@author: jiahao
"""


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("./csv_folder")

#%% merge csv with file name as a new column
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

data = [] # pd.concat takes a list of dataframes as an agrument
for csv in all_filenames:
    frame = pd.read_csv(csv)
    frame['filename'] = os.path.basename(csv)
    data.append(frame)

df = pd.concat(data, ignore_index=True) #dont want pandas to try an align row indexes
df['epoch']=df['epoch']+1
df.to_csv("merged_data.csv", index=False, encoding='utf-8-sig')

#%% keep max f1
idx = df.groupby(['filename'])['valid_instance_f1'].transform(max) == df['valid_instance_f1']
result = df[idx]

result.to_csv("max_f1.csv", index=False)

#%%
df_selected=df[['epoch', 'valid_instance_f1', 'filename']]


#%%


df_selected.filename.replace({
    'PETA_resnet101_20200920-073345.csv': 'ResNet-101',
    'PETA_resnext50_32x4d_20200906-151430.csv': 'ResNeXt-50_32x4d',
    'PETA_dpn131_20200906-115445.csv': 'DPN-131',
    'PETA_resnext101_32x8d_20200906-161038.csv': 'ResNeXt-101_32x8d',
    'PETA_dpn68_20200906-051053.csv': 'DPN-68',
    'PETA_dpn92_20200906-063851.csv': 'DPN-92',
    'PETA_densenet121_20200906-161404.csv': 'DenseNet121',
    'PETA_se_resnet101_20200927-062619.csv': 'SE_ResNet-101',
    'PETA_dpn107_20200906-070344.csv': 'DPN-107',
    'PETA_dpn68b_20200906-053914.csv': 'DPN-68b',
    'PETA_resnet50_20200905-234508.csv': 'ResNet-50',
    'PETA_se_resnet50_20200906-060840.csv': 'SE_ResNet-50'
    },
    inplace=True)

#%%

import seaborn as sns
plt.figure(figsize=(15,8))
# g = sns.lineplot(data=df_selected, x='epoch', y='valid_instance_f1', hue='filename',
#                  hue_order = ['ResNet-50',
#                                 'ResNet-101',
#                                 'DenseNet121',
#                                 'ResNeXt-50_32x4d',
#                                 'ResNeXt-101_32x8d',
#                                 'DPN-68',
#                                 'DPN-68b',
#                                 'DPN-92',
#                                 'DPN-107',
#                                 'DPN-131',
#                                 'SE_ResNet-50',
#                                 'SE_ResNet-101'
#                                 ])

g = sns.lineplot(data=df_selected, x='epoch', y='valid_instance_f1', hue='filename')

g.get_figure().savefig('test.png', dpi=300)

#%%

loss_train = df['train_loss']
loss_val = df['valid_loss']
epochs = df['epoch']
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()