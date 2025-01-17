#!/usr/bin/python3
#\file    maf_std.py
#\brief   Moving average standard deviation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.01, 2021
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  f_base= lambda x: np.tanh(3.-x)

  #noise= 0.0
  #noise= 0.01
  noise= 0.02
  #noise= 0.05
  N_maf= 20

  X= np.linspace(0,5,500)
  Y= []
  Y_std= []
  Y_maf_std= []
  for x in X:
    #Y.append(f_base(x))
    Y.append(f_base(x) + np.abs(np.random.normal(loc=0.5*noise*max(0,x-2),scale=noise*max(0,x-2))))
    Y_std.append(np.std(Y[-N_maf:]))
    Y_maf= [np.mean(Y[max(0,i+1-N_maf//2):i+1+N_maf//2]) for i in range(len(Y))]
    Y_maf_std.append(np.std((np.array(Y)-Y_maf)[-N_maf:]))


  fig= plt.figure()
  ax= fig.add_subplot(1,1,1)
  ax.plot(X, Y, color='blue', linestyle='dotted', label='Original')
  ax.plot(X, Y_maf, color='red',  linestyle='solid', label='MAF')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.legend()

  ax2= ax.twinx()
  ax2.plot(X, Y_std, color='purple',  linestyle='dashdot', label='STD')
  ax2.plot(X, Y_maf_std, color='green',  linestyle='dashed', label='MAF_STD')
  ax2.set_ylabel('std')
  ax2.legend(loc='upper right', bbox_to_anchor=(1.0,0.8))
  plt.show()

