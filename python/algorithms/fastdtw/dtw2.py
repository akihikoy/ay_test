#!/usr/bin/python3
#\file    dtw2.py
#\brief   Applying fastdtw to actual data.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020
# NOTE: Need to install fastdtw by:
#   $ sudo apt-get -f install python3-scipy
#   $ pip install fastdtw
# https://pypi.org/project/fastdtw/
# NOTE: Need to use Python 3 (#!/usr/bin/python3).

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def LoadXFSeq(filename):
  seq= []
  with open(filename,'r') as fp:
    while True:
      line= fp.readline()
      if not line: break
      values= list(map(float,line.split()))
      seq.append([-values[2],values[1]])
  return seq

if __name__=='__main__':
  x_f_seq1= LoadXFSeq('../../data/time_f_z001.dat')
  x_f_seq2= LoadXFSeq('../../data/time_f_z002.dat')

  distance, path = fastdtw(x_f_seq1, x_f_seq2, dist=euclidean)
  print(distance, path)
