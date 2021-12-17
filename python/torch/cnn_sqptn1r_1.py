#!/usr/bin/python3
#\file    cnn_sqptn1r_1.py
#\brief   Learning the square pattern 1r task with CNN on PyTorch.
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

'''
Generate the dataset by:
$ ./gen_sqptn1r.py train
$ ./gen_sqptn1r.py test
'''
class SqPtn1rDataset(torch.utils.data.Dataset):
  def __init__(self, root='data_generated/sqptn1r/', transform=None, train=True):
    self.transform= transform
    self.image_paths= []
    self.imager_paths= []
    self.labels= []
    self.root= root
    self.MakePathLabelList(train)

  def LoadLabel(self, filepath):
    with open(filepath,'r') as fp:
      return float(fp.read().strip())

  def MakePathLabelList(self, train):
    dir_train= 'train' if train else 'test'
    imgext= '.jpg'
    image_paths__labels= [
        (os.path.join(self.root, dir_train, 'images', filename),
         os.path.join(self.root, dir_train, 'images', filename.replace(imgext,'r'+imgext)),
         self.LoadLabel(os.path.join(self.root, dir_train, 'labels', filename+'.dat')) )
        for filename in os.listdir(os.path.join(self.root, dir_train, 'images')) if not filename.endswith('r'+imgext)]
    self.image_paths= [image_path for image_path,imager_path,label in image_paths__labels]
    self.imager_paths= [imager_path for image_path,imager_path,label in image_paths__labels]
    self.labels= [label for image_path,imager_path,label in image_paths__labels]

  def LoadImage(self, image_path):
    with open(image_path,'rb') as f:
      img= PILImage.open(f)
      img= img.convert('RGB')
    return img if self.transform is None else self.transform(img)

  def __getitem__(self, index):
    img= self.LoadImage(self.image_paths[index])
    imgr= self.LoadImage(self.imager_paths[index])
    tr_value= lambda v: torch.autograd.Variable(torch.tensor([v]))
    return img, imgr, tr_value(self.labels[index])

  def __len__(self):
    return len(self.image_paths)
