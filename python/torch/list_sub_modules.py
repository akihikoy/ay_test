#!/usr/bin/python3
#\file    list_sub_modules.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.08, 2022
import torch

def ListChildModules(net):
  def routine(m,indent):
    for subn,subm in m.named_children():
      print('  '*indent+subn)
      #if indent>0: continue
      routine(subm,indent+1)
  routine(net,0)

class TFCN1(torch.nn.Module):
  def __init__(self, p_dropout=0.02):
    super(TFCN1,self).__init__()
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 100),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(100, 100),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(100, 1),
          )
    self.net_fc2= torch.nn.Sequential(
          torch.nn.Linear(1, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 1),
          )
  def forward(self, x):
    x2= self.net_fc1(x)
    y= self.net_fc2(x2)
    return x2,y


if __name__=='__main__':
  net= TFCN1()
  ListChildModules(net)
