# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
I'd like to thank https://github.com/namiyousef/multi-task-learning/blob/main/data/data.py
for their amazing tutorial
"""
import os
import h5py
import pandas as pd
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from config.config import work_dir


class H5FileDataset(Dataset):
    def __init__(self, h5_filename, transform=None, target_transform=None):
        self.h5_filename = h5_filename
        self.img_h5_file = self._load_h5_file(self.h5_filename)
        self.all_labels = self.img_h5_file['labels'][:]


    def __len__(self):
        return len(self.all_labels)


    def __getitem__(self, idx):
        img = self.img_h5_file['img_data'][idx]
        label = self.img_h5_file['labels'][idx]
        # sample = {'data': img,
        #           'label': label,
        #           'img_idx': idx}
        return img, label


    def _load_h5_file(self, h5_filename):
        file = h5py.File(h5_filename, 'r')
        img_data = file['pic_mat']
        img_labels = file['labels']
        return dict(file=file, img_data=img_data, labels=img_labels)


train_dataset = H5FileDataset(r'D:\wliwa\Desktop\FERG_DB_256\h5_files\train_dataset.h5')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
batch_images, batch_labels = next(iter(train_dataloader))
print('Batch size:', batch_images.shape)
plt.figure()
plt.imshow(batch_images[0])
plt.show()


test_dataset = H5FileDataset(r'D:\wliwa\Desktop\FERG_DB_256\h5_files\test_dataset.h5')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True)
batch_images, batch_labels = next(iter(test_dataloader))
print('Batch size:', batch_images.shape)
plt.figure()
plt.imshow(batch_images[0])
plt.show()


kdef_test_dataset = H5FileDataset(r'D:\wliwa\Desktop\KDEF\h5_files\test_dataset.h5')
test_dataloader = torch.utils.data.DataLoader(kdef_test_dataset, batch_size=50, shuffle=True)
batch_images, batch_labels = next(iter(test_dataloader))
print('Batch size:', batch_images.shape)
plt.figure()
plt.imshow(batch_images[0])
plt.show()



