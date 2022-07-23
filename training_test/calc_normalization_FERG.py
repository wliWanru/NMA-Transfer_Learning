# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@foxmail.com
@create: 2022/7/22-23:57 
"""
import os

import h5py
import numpy as np

os.chdir(r'C:\Users\wliwa\Desktop\drive\datasets\FERG_h5\new')

train_file = h5py.File('train_dataset.h5', 'r')
train_mat = train_file['pic_mat'][:]/255
test_file = h5py.File('test_dataset.h5', 'r')
test_mat = test_file['pic_mat'][:]/255

combined_mat = np.concatenate([train_mat, test_mat], axis=0)

mean_rgb = np.mean(combined_mat, axis=(0, 1, 2))
std_rgb = np.std(combined_mat, axis=(0, 1, 2))

train_file.close()
test_file.close()

print(mean_rgb)
print(std_rgb)
