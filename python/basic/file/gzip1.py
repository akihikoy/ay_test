#!/usr/bin/python3
#\file    gzip1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.04, 2020
import gzip
import six.moves.cPickle as pickle
import numpy as np

if __name__=='__main__':
  data= {}
  data['a']= 'test data'
  data['x']= [1,2,3,4,5]
  data['y']= np.array([6.,7.,8.])

  print('Original data:',data)

  filename= '/tmp/pickle1.dat.gz'
  pickle.dump(data,gzip.open(filename, 'w'))
  print('data is dumped into:',filename)

  data2= pickle.load(gzip.open(filename, 'rb'))
  print('Loaded data:',data2)

  print('For each key, the contents are the same?')
  for key in data.keys():
    print(key, type(data[key]), type(data2[key]), data[key] == data2[key])



