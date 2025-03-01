#!/usr/bin/python3
#\file    cnn_sqptn3_1.py
#\brief   Learning the square pattern 3 task with CNN on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.15, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image as PILImage
import os
from lr_sch_2 import ReduceLRAtCondition


#sqptn3= 'sqptn3'
sqptn3= 'sqptn3l'  #Larger dataset.
#A_SIZE1,A_SIZE2= 0.3,0.3
A_SIZE1,A_SIZE2= 0.1,0.1
DATASET_ROOT= 'data_generated/{}/{}_{}/'.format(sqptn3,A_SIZE1,A_SIZE2)
OUTFEAT_SCALE= 1.0/(0.5*(A_SIZE1+A_SIZE2))

'''
Generate the dataset by:
$ ./gen_sqptn3.py train [A_SIZE1] [A_SIZE2]
$ ./gen_sqptn3.py test [A_SIZE1] [A_SIZE2]
'''
class SqPtn3Dataset(torch.utils.data.Dataset):
  def __init__(self, root=DATASET_ROOT, transform=None, train=True):
    self.transform= transform
    self.image1_paths= []
    self.image2_paths= []
    self.in_feats= []
    self.out_feats= []
    self.root= root
    self.MakeIOList(train)

  def LoadValue(self, filepath):
    with open(filepath,'r') as fp:
      return float(fp.read().strip())

  def MakeIOList(self, train):
    dir_train= 'train' if train else 'test'
    image12path_infeat_outfeat= [
        (os.path.join(self.root, dir_train, 'input', filename),
         os.path.join(self.root, dir_train, 'input', filename.replace('-1.jpg','-2.jpg')),
         self.LoadValue(os.path.join(self.root, dir_train, 'input', filename.replace('-1.jpg','.dat'))),
         self.LoadValue(os.path.join(self.root, dir_train, 'output', filename.replace('-1.jpg','.dat'))) )
        for filename in os.listdir(os.path.join(self.root, dir_train, 'input'))
        if filename[-6:]=='-1.jpg']
    self.image1_paths= [image1_path for image1_path,image2_path,in_feat,out_feat in image12path_infeat_outfeat]
    self.image2_paths= [image2_path for image1_path,image2_path,in_feat,out_feat in image12path_infeat_outfeat]
    self.in_feats= [in_feat for image1_path,image2_path,in_feat,out_feat in image12path_infeat_outfeat]
    self.out_feats= [out_feat*OUTFEAT_SCALE for image1_path,image2_path,in_feat,out_feat in image12path_infeat_outfeat]

  def __getitem__(self, index):
    with open(self.image1_paths[index],'rb') as f:
      img1= PILImage.open(f)
      img1= img1.convert('RGB')
    img1= img1 if self.transform is None else self.transform(img1)
    with open(self.image2_paths[index],'rb') as f:
      img2= PILImage.open(f)
      img2= img2.convert('RGB')
    img2= img2 if self.transform is None else self.transform(img2)
    tr_value= lambda v: torch.autograd.Variable(torch.tensor([v]))
    return img1, img2, tr_value(self.in_feats[index]), tr_value(self.out_feats[index])

  def __len__(self):
    return len(self.image1_paths)

def GetDataTransforms(mode):
  if mode=='train':
    return torchvision.transforms.Compose([
        #torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
  if mode=='eval':
    return torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
  if mode=='none':
    return torchvision.transforms.Compose([
        #torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
      ])

class TCNN3(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN3,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

'''
class TCNN3a(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN3a,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 256),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(256, 256),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 256),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(256, 256),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 256),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(256, 256),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(256+256+256, 256),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(256, 256),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(256, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3b(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.001, p_dropout_cnn=0.0):
    super(TCNN3b,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout_cnn),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)
'''

class TCNN3c(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN3c,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3d(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN3d,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    n_fc= 4096
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3e(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN3e,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    n_fc= 4096
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(n_fc, n_fc),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3f(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=6000):
    super(TCNN3f,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3f2(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=6000):
    super(TCNN3f2,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3f2a(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=3000):
    super(TCNN3f2a,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3f3(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=6000):
    super(TCNN3f3,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.Tanh(),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN3g(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=3000):
    super(TCNN3g,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, n_fc//2),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc//2+n_fc*2, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN4(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN4,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_imgc= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_imgc(self.net_img1(torch.FloatTensor(*((1,)+img1_shape)))).view(1,-1).shape[1]
    n_img2_out= self.net_imgc(self.net_img2(torch.FloatTensor(*((1,)+img2_shape)))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    #self.net_fc2x= torch.nn.Sequential(
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_imgc(self.net_img1(x1))
    x1= x1.view(x1.size(0), -1)
    x1= (self.net_fc2a(x1))
    x2= self.net_imgc(self.net_img2(x2))
    x2= x2.view(x2.size(0), -1)
    x2= (self.net_fc2b(x2))
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN4a(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN4a,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_imgc= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_imgc(self.net_img1(torch.FloatTensor(*((1,)+img1_shape)))).view(1,-1).shape[1]
    n_img2_out= self.net_imgc(self.net_img2(torch.FloatTensor(*((1,)+img2_shape)))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    #self.net_fc2x= torch.nn.Sequential(
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_imgc(self.net_img1(x1))
    x1= x1.view(x1.size(0), -1)
    x1= (self.net_fc2a(x1))
    x2= self.net_imgc(self.net_img2(x2))
    x2= x2.view(x2.size(0), -1)
    x2= (self.net_fc2b(x2))
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN4b(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TCNN4b,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_imgc= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_imgc(self.net_img1(torch.FloatTensor(*((1,)+img1_shape)))).view(1,-1).shape[1]
    n_img2_out= self.net_imgc(self.net_img2(torch.FloatTensor(*((1,)+img2_shape)))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          )
    self.net_fc2= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out+n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    #self.net_fc2a= torch.nn.Sequential(
          #torch.nn.Linear(n_img1_out, 2048),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    #self.net_fc2b= torch.nn.Sequential(
          #torch.nn.Linear(n_img2_out, 2048),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    #self.net_fc2x= torch.nn.Sequential(
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_imgc(self.net_img1(x1))
    x1= x1.view(x1.size(0), -1)
    #x1= (self.net_fc2a(x1))
    x2= self.net_imgc(self.net_img2(x2))
    x2= x2.view(x2.size(0), -1)
    #x2= (self.net_fc2b(x2))
    x= self.net_fc2(torch.cat((x1,x2),1))
    y= self.net_fc1(y)
    x= torch.cat((x,y),1)
    return self.net_fc3(x)

class TAlexCNN3(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02):
    super(TAlexCNN3,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048+2048+2048, 2048),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(2048, 2048),
          #torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= self.net_fc1(y)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN5(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=3000):
    super(TCNN5,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.n_fc= n_fc
    #self.net_fc1= torch.nn.Sequential(
          #torch.nn.Linear(1, n_fc),
          #torch.nn.LeakyReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*3, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    #y= self.net_fc1(y)
    y= y.repeat(1, self.n_fc)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)

class TCNN5a(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_fc=3000):
    super(TCNN5a,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.n_fc= n_fc
    #self.net_fc1= torch.nn.Sequential(
          #torch.nn.Linear(1, n_fc),
          #torch.nn.LeakyReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #)
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_fc*2+1, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    #y= self.net_fc1(y)
    #y= y.repeat(1, self.n_fc)
    x= torch.cat((x1,x2,y),1)
    return self.net_fc3(x)


if __name__=='__main__':
  import sys
  initial_model_file= sys.argv[1] if len(sys.argv)>1 else None

  dataset_train= SqPtn3Dataset(transform=GetDataTransforms('train'), train=True)
  dataset_test= SqPtn3Dataset(transform=GetDataTransforms('eval'), train=False)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input img1 type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] input img2 type, shape:',type(dataset_train[0][1]),dataset_train[0][1].shape)
  print('dataset_train[0] input feat value:',dataset_train[0][2])
  print('dataset_train[0] output feat value:',dataset_train[0][3])
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input img1 type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] input img2 type, shape:',type(dataset_test[0][1]),dataset_test[0][1].shape)
  print('dataset_test[0] input feat value:',dataset_test[0][2])
  print('dataset_test[0] output feat value:',dataset_test[0][3])
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img1,img2,in_feat,out_feat= dataset_train[i_data]
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/in={1:.3f}\nout={2:.3f}'.format(i_data,in_feat,out_feat), fontsize=10)
    ax.imshow(img.permute(1,2,0))
  fig.tight_layout()
  plt.show()
  '''

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  img1_shape,img2_shape= dataset_train[0][0].shape,dataset_train[0][1].shape
  #net= TCNN3(img1_shape,img2_shape)
  ##net= TCNN3a(img1_shape,img2_shape)
  ##net= TCNN3b(img1_shape,img2_shape)
  #net= TCNN3c(img1_shape,img2_shape)
  #net= TCNN3d(img1_shape,img2_shape)
  #net= TCNN3e(img1_shape,img2_shape)
  #net= TCNN3e(img1_shape,img2_shape,p_dropout=0.1)
  #net= TCNN3f(img1_shape,img2_shape)
  #net= TCNN3f(img1_shape,img2_shape, n_fc=3000)
  #net= TCNN3f2(img1_shape,img2_shape, n_fc=3000)
  #net= TCNN3f2(img1_shape,img2_shape, n_fc=4000)
  #net= TCNN3f3(img1_shape,img2_shape, n_fc=3000)
  #net= TCNN3f2a(img1_shape,img2_shape, n_fc=3000)
  #net= TCNN3g(img1_shape,img2_shape)
  #net= TCNN4(img1_shape,img2_shape)
  #net= TCNN4a(img1_shape,img2_shape)
  #net= TCNN4b(img1_shape,img2_shape)
  #net= TAlexCNN3(img1_shape,img2_shape)
  net= TCNN5(img1_shape,img2_shape)
  #net= TCNN5(img1_shape,img2_shape, n_fc=500)
  #net= TCNN5a(img1_shape,img2_shape)

  #NOTE: Switch the device.
  #device= 'cpu'
  device= 'cuda'
  #device= 'cuda:1'  #Second GPU.
  if device=='cuda' and not torch.cuda.is_available():
    device= 'cpu'
    print('device is modified to cpu since cuda is not available.')
  net= net.to(device)

  if initial_model_file is not None:
    net.load_state_dict(torch.load(initial_model_file, map_location=device))

  print(net)

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  sch= None
  swa_sch= None
  ##opt= torch.optim.Adam(net.parameters(), lr=0.001)
  ###opt= torch.optim.SGD(net.parameters(), lr=0.004)
  ###opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=5e-4)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.01)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #opt= torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.95, weight_decay=5e-4)
  ##opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-8)
  ###opt= torch.optim.Adagrad(net.parameters())
  ###opt= torch.optim.RMSprop(net.parameters())
  #TEST: Learning rate scheduler:
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  #opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95, weight_decay=0.0)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  ##sch= torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
  #opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=100, threshold=0.0001, threshold_mode='rel', verbose=True)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        #mode='min', factor=0.5, patience=100, cooldown=100,
        #threshold=-0.001, threshold_mode='rel',
        #verbose=True)
  opt= torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.95, weight_decay=0.005)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  sch= ReduceLRAtCondition(opt,
         mode='gt', factor=0.5, patience=10, cooldown=200, threshold=0.001,
         verbose=True)
  #sch= torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0001, max_lr=0.1, step_size_up=100, mode='triangular2', verbose=True)
  #TEST: Stochastic Weight Averaging:
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #swa_model= torch.optim.swa_utils.AveragedModel(net)
  #sch= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
  #swa_start= 160
  #swa_sch= torch.optim.swa_utils.SWALR(opt, swa_lr=0.05)

  loss= torch.nn.MSELoss()

  #opt= torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.95, weight_decay=0.0001)
  #loss= torch.nn.L1Loss()

  #opt= torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  #loss= torch.nn.HuberLoss(reduction='mean', delta=0.1*OUTFEAT_SCALE)

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 40
  N_epoch= 3000

  #Save the info into a file.
  now= time.localtime()
  time_stamp= '%04i.%02i.%02i-%02i.%02i.%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  info_filename= 'data_generated/log/{}-{}-{}-{}.info'.format(sqptn3,net.__class__.__name__,opt.__class__.__name__,time_stamp)
  with open(info_filename,'w') as fp:
    fp.write('net:\n'+str(net)+'\n\n')
    fp.write('opt:\n'+str(opt)+'\n\n')
    fp.write('sch:\n'+str(sch)+'\n'+str(sch.state_dict())+'\n\n')
    fp.write('loss:\n'+str(loss)+'\n'+str(loss.state_dict())+'\n\n')
    fp.write('dataset_train: {} {}\n\n'.format(len(dataset_train),dataset_train.root))
    fp.write('dataset_test: {} {}\n\n'.format(len(dataset_test),dataset_test.root))
    fp.write('N_epoch,N_batch: {}, {}\n\n'.format(N_epoch,N_batch))
  print('saved info into:',info_filename)

  loader_train= torch.utils.data.DataLoader(
                  dataset=dataset_train,
                  batch_size=N_batch,
                  shuffle=True,
                  num_workers=2)
  loader_test= torch.utils.data.DataLoader(
                  dataset=dataset_test,
                  batch_size=N_batch,
                  shuffle=False,
                  num_workers=2)

  best_net_state= None
  best_net_loss= None
  log_train_time= []
  log_test_time= []
  log_loss_per_epoch= []
  log_loss_test_per_epoch= []
  log_lr= []
  for i_epoch in range(N_epoch):
    log_loss_per_epoch.append(0.0)
    log_train_time.append(time.time())
    log_lr.append(opt.param_groups[0]['lr'])
    net.train()  # training mode; using dropout.
    for i_step, (batch_imgs1, batch_imgs2, batch_infeats, batch_outfeats) in enumerate(loader_train):
      #torch.autograd.Variable()
      b_imgs1= batch_imgs1
      b_imgs2= batch_imgs2
      b_infeats= batch_infeats
      b_outfeats= batch_outfeats
      b_imgs1,b_imgs2,b_infeats,b_outfeats= b_imgs1.to(device),b_imgs2.to(device),b_infeats.to(device),b_outfeats.to(device)

      opt.zero_grad()
      pred= net(b_imgs1,b_imgs2,b_infeats)
      err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)
      #err= loss(pred, b_outfeats)*2.0

      err.backward()
      opt.step()
      log_loss_per_epoch[-1]+= err.item()/len(loader_train)
      #print(i_epoch,i_step,err)
    log_train_time[-1]= time.time()-log_train_time[-1]

    #Test the network with the test data.
    log_loss_test_per_epoch.append(0.0)
    log_test_time.append(time.time())
    mse= 0.0  #MSE test.
    net.eval()  # evaluation mode; disabling dropout.
    with torch.no_grad():  # suppress calculating gradients.
      for i_step, (batch_imgs1, batch_imgs2, batch_infeats, batch_outfeats) in enumerate(loader_test):
        b_imgs1= batch_imgs1
        b_imgs2= batch_imgs2
        b_infeats= batch_infeats
        b_outfeats= batch_outfeats
        b_imgs1,b_imgs2,b_infeats,b_outfeats= b_imgs1.to(device),b_imgs2.to(device),b_infeats.to(device),b_outfeats.to(device)
        if swa_sch is None or i_epoch!=N_epoch-1:
          pred= net(b_imgs1,b_imgs2,b_infeats)
        else:
          pred= swa_model(b_imgs1,b_imgs2,b_infeats)
        err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)
        log_loss_test_per_epoch[-1]+= err.item()/len(loader_test)
        mse+= torch.mean((pred-b_outfeats)**2).item()/len(loader_test)
        #print(i_epoch,i_step,err)
    log_test_time[-1]= time.time()-log_test_time[-1]
    if best_net_state is None or log_loss_test_per_epoch[-1]<best_net_loss:
      best_net_state= copy.deepcopy(net.state_dict())
      best_net_loss= log_loss_test_per_epoch[-1]
    if sch is not None:
      if swa_sch is None or i_epoch<=swa_start:
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
          #sch.step(log_loss_per_epoch[-1])
          sch.step(log_loss_test_per_epoch[-1])
          #sch.step(best_net_loss)
        elif isinstance(sch, ReduceLRAtCondition):
          N_maf= 20
          #schmetric= np.std(log_loss_test_per_epoch[-N_maf:])
          maf= [np.mean(log_loss_test_per_epoch[max(0,i+1-N_maf//2):i+1+N_maf//2]) for i in range(max(0,len(log_loss_test_per_epoch)-N_maf), len(log_loss_test_per_epoch))]
          schmetric= np.std((np.array(log_loss_test_per_epoch)[-len(maf):]-maf))
          sch.step(schmetric)
        else:
          sch.step()
      elif swa_sch is not None and i_epoch>swa_start:
        swa_model.update_parameters(net)
        swa_sch.step()
        if i_epoch==N_epoch-1:
          torch.optim.swa_utils.update_bn(loader_train, swa_model)
    print(i_epoch,log_loss_per_epoch[-1],log_loss_test_per_epoch[-1],mse)
  print('training time:',np.sum(log_train_time))
  print('testing time:',np.sum(log_test_time))
  print('best loss:',best_net_loss)

  #Recall the best net parameters:
  net.load_state_dict(best_net_state)

  #Save the model parameters into a file.
  #To load it: net.load_state_dict(torch.load(FILEPATH))
  torch.save(net.state_dict(), 'model_learned/cnn_{}_1-{}_{}.pt'.format(sqptn3,A_SIZE1,A_SIZE2))

  #Save the log into a file.
  log_filename= 'data_generated/log/{}-{}-{}-{}.log'.format(sqptn3,net.__class__.__name__,opt.__class__.__name__,time_stamp)
  with open(log_filename,'w') as fp:
    fp.write('#i_epoch train_time test_time loss_train loss_test\n')
    for i_epoch, (train_time, test_time, loss_train, loss_test, lr) in enumerate(zip(log_train_time, log_test_time, log_loss_per_epoch, log_loss_test_per_epoch, log_lr)):
      fp.write('{} {} {} {} {} {}\n'.format(i_epoch, train_time, test_time, loss_train, loss_test, lr))
  print('saved info into:',info_filename)
  print('saved log into:',log_filename)

  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  fig1.tight_layout()

  net.eval()  # evaluation mode; disabling dropout.
  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_test)))
    img1,img2,in_feat,out_feat= dataset_test[i_data]
    pred= net(img1.view((1,)+img1.shape).to(device),img2.view((1,)+img2.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device)).data.cpu().item()/OUTFEAT_SCALE
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}/in={1:.3f}\nout={2:.3f}\n/pred={3:.3f}'.format(i_data,in_feat,out_feat,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  fig3= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img1,img2,in_feat,out_feat= dataset_train[i_data]
    pred= net(img1.view((1,)+img1.shape).to(device),img2.view((1,)+img2.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device)).data.cpu().item()/OUTFEAT_SCALE
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    ax= fig3.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/in={1:.3f}\nout={2:.3f}\n/pred={3:.3f}'.format(i_data,in_feat,out_feat,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig3.tight_layout()

  plt.show()
  #'''
