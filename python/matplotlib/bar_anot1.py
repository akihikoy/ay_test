#!/usr/bin/python3
#\file    bar_anot1.py
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
      plt.annotate('{0}={1}'.format('value',value),
                xy=(rect.get_x()+rect.get_width()/2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend(loc='upper left')
  plt.show()
