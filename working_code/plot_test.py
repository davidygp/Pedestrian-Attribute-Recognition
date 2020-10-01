#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:15:19 2020

@author: jiahao
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/jiahao/Downloads/20200830-073236.csv')

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
