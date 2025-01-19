#!/usr/bin/python3
#\file    py2py3_numdiff.py
#\brief   Difference of numerical computation between Py2 and Py3;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.20, 2025
from __future__ import print_function
import numpy as np
import platform

def Main():
  p1= [-0.027657739150557383, 0.039555240822999965]
  base= [[-0.04296878500756207, -0.0015296035083907015], p1]
  sample= [[-0.04157682596197626, 0.003141223794586525], p1, [-0.02599420767177901, 0.04080823347131331]]
  base= np.array(base)
  sample= np.array(sample)
  #base= np.array(base, dtype=np.float128)
  #sample= np.array(sample, dtype=np.float128)
  h, t = base
  dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))

  print('Python version=',platform.python_version())
  print('dists=',dists.tolist())


if __name__=='__main__':
  Main()
