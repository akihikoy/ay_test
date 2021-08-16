#!/usr/bin/python
#\file    bar_anot2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  values= [2,1,3]
  rects= plt.bar([1,2,3], values, 0.2, yerr=[0.2,0.2,0.1], label='data')
  for rect,value in zip(rects,values):
      plt.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
              '{0}={1}'.format('value',value),
              {'ha':'center', 'va':'bottom'})
  for rect,value in zip(rects,values):
      plt.text(rect.get_x()+rect.get_width()/2, rect.get_height()/2,
              '{0}={1}'.format('VALUE',value),
              {'ha':'center', 'va':'center'}, rotation=90)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend(loc='upper left')
  plt.show()
