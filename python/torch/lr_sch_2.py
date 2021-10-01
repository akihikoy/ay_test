#!/usr/bin/python3
#\file    lr_sch_2.py
#\brief   Test learning schedulers of PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.01, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import sys

class ReduceLRAtCondition(object):
  def __init__(self, optimizer, mode='gt', factor=0.1, patience=10, cooldown=0,
         threshold=1e-4, min_lr=0, eps=1e-8, verbose=False):

    if factor >= 1.0:
      raise ValueError('Factor should be < 1.0.')
    self.factor = factor

    # Attach optimizer
    if not isinstance(optimizer, torch.optim.Optimizer):
      raise TypeError('{} is not an Optimizer'.format(
        type(optimizer).__name__))
    self.optimizer = optimizer

    if isinstance(min_lr, list) or isinstance(min_lr, tuple):
      if len(min_lr) != len(optimizer.param_groups):
        raise ValueError("expected {} min_lrs, got {}".format(
          len(optimizer.param_groups), len(min_lr)))
      self.min_lrs = list(min_lr)
    else:
      self.min_lrs = [min_lr] * len(optimizer.param_groups)

    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0
    self.mode = mode
    self.threshold = threshold
    self.num_satisfied_epochs = None
    self.eps = eps
    self.last_epoch = 0
    self._reset()

  def _reset(self):
    #self.best = self.mode_worse
    self.cooldown_counter = 0
    self.num_satisfied_epochs = 0

  def step(self, metrics):
    # convert `metrics` to float, in case it's a zero-dim Tensor
    current = float(metrics)
    self.last_epoch = self.last_epoch + 1

    if self.is_satisfied(current):
      self.num_satisfied_epochs += 1
    else:
      self.num_satisfied_epochs = 0

    if self.in_cooldown():
      self.cooldown_counter -= 1
      self.num_satisfied_epochs = 0

    if self.num_satisfied_epochs > self.patience:
      self._reduce_lr(self.last_epoch)
      self.cooldown_counter = self.cooldown
      self.num_satisfied_epochs = 0

    self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

  def _reduce_lr(self, epoch):
    for i, param_group in enumerate(self.optimizer.param_groups):
      old_lr = float(param_group['lr'])
      new_lr = max(old_lr * self.factor, self.min_lrs[i])
      if old_lr - new_lr > self.eps:
        param_group['lr'] = new_lr
        if self.verbose:
          print('Epoch {:5d}: reducing learning rate'
              ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

  def in_cooldown(self):
    return self.cooldown_counter > 0

  def is_satisfied(self, a):
    if self.mode == 'gt':
      return a > self.threshold

    elif self.mode == 'lt':
      return a < self.threshold

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

if __name__=='__main__':
  log_file_name= sys.argv[1]

  with open(log_file_name,'r') as fp:
    data_train_time= []
    data_test_time= []
    data_loss_per_epoch= []
    data_loss_test_per_epoch= []
    while True:
      line= fp.readline()
      if not line:  break
      if line[0]=='#':  continue
      i_epoch,train_time,test_time,loss_train,loss_test= map(float,line.split()[:5])
      data_train_time.append(train_time)
      data_test_time.append(test_time)
      data_loss_per_epoch.append(loss_train)
      data_loss_test_per_epoch.append(loss_test)

  net= torch.nn.Sequential(torch.nn.Linear(1, 200))
  opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
         #mode='min', factor=0.5, patience=100, cooldown=100,
         ##mode='min', factor=0.5, patience=50, cooldown=50,
         ##threshold=0.0001, threshold_mode='rel',
         #threshold=-0.001, threshold_mode='rel',
         ##threshold=-0.01, threshold_mode='rel',
         #verbose=True)
  sch= ReduceLRAtCondition(opt,
         #mode='gt', factor=0.5, patience=10, cooldown=200, threshold=0.001,
         mode='gt', factor=0.5, patience=10, cooldown=100, threshold=0.03,
         verbose=True)

  log_loss_per_epoch= []
  log_loss_test_per_epoch= []
  log_schmetric= []
  log_lr= []
  for i_epoch,(loss_train, loss_test) in enumerate(zip(data_loss_per_epoch,data_loss_test_per_epoch)):
    log_loss_per_epoch.append(loss_train)
    log_loss_test_per_epoch.append(loss_test)
    if isinstance(sch,torch.optim.lr_scheduler.ReduceLROnPlateau):
      schmetric= log_loss_test_per_epoch[-1]
      sch.step(schmetric)
    elif isinstance(sch,ReduceLRAtCondition):
      N_maf= 20
      #schmetric= np.std(log_loss_test_per_epoch[-N_maf:])
      maf= [np.mean(log_loss_test_per_epoch[max(0,i+1-N_maf//2):i+1+N_maf//2]) for i in range(max(0,len(log_loss_test_per_epoch)-N_maf), len(log_loss_test_per_epoch))]
      schmetric= np.std((np.array(log_loss_test_per_epoch)[-len(maf):]-maf))
      sch.step(schmetric)
    else:
      schmetric= 0
      sch.step()
    log_schmetric.append(schmetric)
    log_lr.append(float(opt.param_groups[0]['lr']))

  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.plot(range(len(log_schmetric)), log_schmetric, color='red', linestyle='dashed', label='std[loss_test]')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend(loc='upper right')
  ax_lr= ax_lc.twinx()
  ax_lr.plot(range(len(log_lr)), log_lr, color='green', label='LR')
  #ax_lr.set_title('Learning rate')
  ax_lr.set_xlabel('epoch')
  ax_lr.set_ylabel('Learning rate')
  #ax_lr.set_yscale('log')
  ax_lr.legend(loc='upper right', bbox_to_anchor=(1.0,0.8))
  fig1.tight_layout()

  plt.show()
