#!/usr/bin/python3
#\file    list_cuda.py
#\brief   List available CUDA devices.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.01, 2021
import torch

if __name__=='__main__':
  print('Number of CUDA devices:', torch.cuda.device_count())
  for i in range(torch.cuda.device_count()):
    print('  cuda:{}: {}'.format(i,torch.cuda.get_device_name('cuda:{}'.format(i))))

