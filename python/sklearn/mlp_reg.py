#!/usr/bin/python3
#\file    mlp_reg.py
#\brief   sklearn MLP for regression test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2024
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def f(x):
  return x*np.sin(3.0*x)

if __name__=='__main__':
  data_x= np.random.uniform(-2.,2.,200).reshape(-1,1)
  data_y= f(data_x)+np.random.uniform(-0.1,0.1,len(data_x)).reshape(-1,1)

  mlp= MLPRegressor(activation='relu', hidden_layer_sizes=(20, 20), solver='adam', tol=1e-6, alpha=0.001, max_iter=200000, random_state=2)
  reg= mlp.fit(data_x, data_y)

  lin= np.linspace(np.min(data_x),np.max(data_x),200).reshape(-1,1)
  y_pred= reg.predict(lin)

  fig= plt.figure(figsize=(8,8))
  ax1= fig.add_subplot(1,1,1,title='Prediction',xlabel='x',ylabel='y')
  ax1.plot(lin, y_pred, color='blue', label='Prediction')
  ax1.scatter(data_x, data_y, marker='+', s=30, color='red', label='Data')
  ax1.legend()

  fig.tight_layout()
  plt.show()
