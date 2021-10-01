#!/usr/bin/python
#\file    maf1.py
#\brief   Moving average filter.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.01, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  f_base= lambda x: np.tanh(5.-x)

  N_maf= 20

  X= np.linspace(0,5,500)
  Y= []
  #Y_maf= []
  for x in X:
    Y.append(f_base(x) + np.abs(np.random.normal(loc=0.01*x,scale=0.02*x)))
    #Y_maf.append(np.mean(Y[-N_maf:]))  #NOTE: Method-1 to calculate MAF.

  #NOTE: Method-2 to calculate MAF (result==Method-1).
  #Y_maf= [np.mean(Y[max(0,i+1-N_maf):i+1]) for i in range(len(Y))]

  #NOTE: Method-1 to calculate MAF centered at the current.
  Y_maf= [np.mean(Y[max(0,i+1-N_maf//2):i+1+N_maf//2]) for i in range(len(Y))]

  fig= plt.figure()
  ax= fig.add_subplot(1,1,1)
  ax.plot(X, Y, color='blue', linestyle='dotted', label='Original')
  ax.plot(X, Y_maf, color='red',  linestyle='solid', label='MAF')
  #ax.plot(x,norm.pdf(x, loc=0.0, scale=0.25), color='black', linestyle='dashdot')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.legend()
  plt.show()

