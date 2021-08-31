#!/usr/bin/python
#\file    scipy_lwr1.py
#\brief   Solving for LWR(x)==y wrt x.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.31, 2021
from lwr_incr5 import *
import matplotlib.pyplot as plt
import scipy.optimize

if __name__=='__main__':
  x_max= -0.06
  y_err= 10.0
  data_x= np.linspace(0.,x_max,10).reshape(-1,1)
  data_y= 20.*(np.pi*0.5+np.arctan(-100.*(data_x-x_max*0.5)))
  data_y+= np.abs(data_x)/np.abs(x_max) * np.random.uniform(-y_err,y_err,size=(len(data_x),1))  #Error

  options= {}
  options['kernel']= 'maxg'
  options['c_min']= 0.01
  options['f_reg']= 0.0001
  lwr= TLWR()
  lwr.Load({'options':options})
  lwr.Init()
  #lwr.UpdateBatch(data_x.tolist(), data_y.tolist())
  for x,y in zip(data_x,data_y):
    lwr.Update(x,y)

  f_opt= lambda y_trg: scipy.optimize.minimize_scalar(lambda x:np.asscalar((lwr.Predict([x],with_var=False,with_grad=False).Y-y_trg)**2), bounds=(x_max,0.0), method='bounded').x
  y_trgs= np.random.uniform(np.min(data_y)-10, np.max(data_y)+10, size=(10,1))
  x_opts= [f_opt(y) for y in y_trgs]
  y_opts= [np.asscalar(lwr.Predict([x],with_var=False,with_grad=False).Y) for x in x_opts]

  pred_x= np.linspace(0.,x_max,100).reshape(-1,1)
  pred= [lwr.Predict(x,with_var=True,with_grad=True) for x in pred_x]
  pred_y= np.array([np.asarray(p.Y).ravel() for p in pred])
  std_y= np.sqrt([np.asarray(p.Var).ravel() for p in pred])
  #print pred_x
  #print [x for x in pred_x]
  #print lwr.X
  #print lwr.Y
  #print pred
  #print pred_y
  #print std_y

  plt.fill_between(pred_x.reshape(-1), (pred_y-std_y).reshape(-1), (pred_y+std_y).reshape(-1), alpha=0.2)
  plt.plot(pred_x, pred_y, color='blue', linewidth=1, label='LWR-mean$\pm$1sd')
  plt.scatter(data_x, data_y, color='red', label='Samples')
  plt.scatter(x_opts, y_trgs, color='orange', marker='x', s=64, label='Target')
  plt.scatter(x_opts, y_opts, color='green', marker='*', s=64, label='Optimized')
  plt.title('LWR')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim(np.max(data_x), np.min(data_x))
  plt.legend(loc='upper left')
  plt.rcParams['keymap.quit'].append('q')
  plt.show()
