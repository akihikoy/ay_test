#!/usr/bin/python3
#\file    mlp_cls.py
#\brief   sklearn MLP for classification test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.15, 2024
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def f(x):
  return x*np.sin(3.0*x)

if __name__=='__main__':
  data_x= np.random.uniform(-2.,2.,200).reshape(-1,1)
  data_y= f(data_x)+np.random.uniform(-0.1,0.1,len(data_x)).reshape(-1,1)
  data_c= (np.abs(data_x)<1).astype(np.int32)

  mlp= MLPClassifier(activation='relu', hidden_layer_sizes=(20, 20), solver='adam', tol=1e-6, alpha=0.001, max_iter=200000, random_state=2)
  cls= mlp.fit(data_x, data_c)

  c_pred= cls.predict(data_x)

  fig= plt.figure(figsize=(8,8))
  ax1= fig.add_subplot(1,1,1,title='Prediction',xlabel='x',ylabel='y')
  ax1.scatter(data_x, data_y, marker='+', s=30, color='green', label='Data')
  ax1.scatter(data_x[data_c.astype(bool)], data_y[data_c.astype(bool)], marker='o', s=30, color='blue', label='Data/True')
  ax1.scatter(data_x[np.logical_not(data_c.astype(bool))], data_y[np.logical_not(data_c.astype(bool))], marker='x', s=30, color='blue', label='Data/False')
  ax1.scatter(data_x[c_pred.astype(bool)], data_y[c_pred.astype(bool)], marker='o', s=10, color='red', label='Pred/True')
  ax1.scatter(data_x[np.logical_not(c_pred.astype(bool))], data_y[np.logical_not(c_pred.astype(bool))], marker='x', s=30, color='red', label='Pred/False')
  ax1.legend()

  fig.tight_layout()
  plt.show()
