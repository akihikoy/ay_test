#!/usr/bin/python3
#\file    func_min1.py
#\brief   Minimizing a function.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.18, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from funcs_2d import Func
import functools

class TMinimizer(torch.nn.Module):
  def __init__(self, func, n_particle=10):
    super(TMinimizer,self).__init__()
    self.x= torch.nn.Parameter( torch.zeros(n_particle,2).normal_(0.,1.) )
    self.func= func
  def forward(self):
    x= self.func(self.x)
    return x

if __name__=='__main__':
  import sys
  xmin= [-1.,-1.]
  xmax= [2.,3.]
  fkind= int(sys.argv[1]) if len(sys.argv)>1 else 1
  func_torch= functools.partial(Func, kind=fkind, env=torch)
  func_numpy= functools.partial(Func, kind=fkind, env=np)

  m= TMinimizer(func_torch, n_particle=1000)
  opt= torch.optim.Adam(m.parameters(), lr=0.2)
  N_epoch= 50

  t_0= time.time()
  m.train()  # training mode; using dropout.
  log_loss_per_epoch= []
  log_best_loss_per_epoch= []
  for i_epoch in range(N_epoch):
    opt.zero_grad()  # clear gradients for next train
    l= m()
    err= torch.mean(l)
    err.backward()
    opt.step()
    log_loss_per_epoch.append(err.item())
    log_best_loss_per_epoch.append(torch.min(l))
  print(f'result: {m.x}')
  print(f'optimization time: {time.time()-t_0}')
  print(f'last loss: {log_loss_per_epoch[-1]}')
  print(f'last best-loss: {log_best_loss_per_epoch[-1]}')

  fig= plt.figure(figsize=(10,5))
  ax_lc= fig.add_subplot(1,2,1)

  ax= fig.add_subplot(1,2,2,projection='3d')
  true_x= np.mgrid[xmin[0]:xmax[0]:(xmax[0]-xmin[0])/100, xmin[1]:xmax[1]:(xmax[1]-xmin[1])/100]
  ax.plot_wireframe(true_x[0], true_x[1], func_numpy(true_x[:,:,:].reshape(2,-1).T).reshape(true_x.shape[1:]), color='green', linewidth=1, label='true_func')
  with torch.no_grad():
    ax.scatter(m.x[:,0], m.x[:,1], func_torch(m.x), marker='*', color=[1,0,0])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss')
  ax_lc.plot(range(len(log_best_loss_per_epoch)), log_best_loss_per_epoch, color='red', label='best loss')
  ax_lc.set_title('Learning curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  #ax_lc.set_yscale('log')
  ax_lc.legend()

  plt.show()
