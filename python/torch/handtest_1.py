#!/usr/bin/python3
#\file    handtest_1.py
#\brief   Handtest dataset for PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.17, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image as PILImage
import os

class HandtestDataset(torch.utils.data.Dataset):
  def __init__(self, root='data_downloaded/handtest/', transform=None, train=True, split_seed=42, test_ratio=0.2):
    self.transform= transform
    self.image_paths= []
    self.root= root
    self.split_seed= split_seed
    self.test_ratio= test_ratio
    self.MakePathList(train)

  def MakePathList(self, train):
    image_paths= np.array([os.path.join(self.root, filename)
                           for filename in os.listdir(self.root)])
    idxes= np.random.RandomState(seed=self.split_seed).permutation(len(image_paths))
    N_train= round(len(image_paths)*(1.-self.test_ratio))
    if train:  self.image_paths= image_paths[idxes[:N_train]]
    else:      self.image_paths= image_paths[idxes[N_train:]]

  def LoadImage(self, image_path):
    with open(image_path,'rb') as f:
      img= PILImage.open(f)
      img= img.convert('RGB')
    return img if self.transform is None else self.transform(img)

  def __getitem__(self, index):
    img= self.LoadImage(self.image_paths[index])
    return (img,)

  def __len__(self):
    return len(self.image_paths)
