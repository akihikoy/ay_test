#!/usr/bin/python3
#\file    cuda_avail.py
#\brief   Check if CUDA is available.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import torch

if __name__=='__main__':
  print('CUDA available?', torch.cuda.is_available())
