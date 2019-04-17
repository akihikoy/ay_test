#!/usr/bin/python
#\file    np_array_idx.py
#\brief   Examples of numpy array slicing and indexing
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.04, 2016
# cf. https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

import numpy as np

def do(expr):
  if '=' not in expr:
    print expr,'=\n',repr(eval(expr))
  else:
    print expr
    exec(expr)

a= np.array([1,2,3,4,5,6,7,8,9])
do('a')

do('a[[0,1,2]]')

do('a[[1,2,5]]= [10,11,12]')
do('a')

m=np.array([[1,2,3],[4,5,6],[7,8,9]])
do('m')

do('m[[0,2],[0,2]]')
do('m[:,[0,2]]')

do('m[[0,0,2,2],[0,2,0,2]]')
do('m[[0,0,2,2],[0,2,0,2]].reshape(2,2)')
do('m[[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]]')
do('m[[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]].reshape(3,3)')
do('m[np.ix_([0,2],[0,2])]')
do('m[np.ix_([0,2],[1,0,2])]')

