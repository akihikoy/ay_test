#!/usr/bin/python3
#\file    lr_sch_1.py
#\brief   Test learning schedulers of PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.30, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import sys

if __name__=='__main__':
  log_file_name= sys.argv[1]

  with open(log_file_name,'r') as fp:
    log_train_time= []
    log_test_time= []
    log_loss_per_epoch= []
    log_loss_test_per_epoch= []
    while True:
      line= fp.readline()
      if not line:  break
      if line[0]=='#':  continue
      i_epoch,train_time,test_time,loss_train,loss_test= map(float,line.split()[:5])
      log_train_time.append(train_time)
      log_test_time.append(test_time)
      log_loss_per_epoch.append(loss_train)
      log_loss_test_per_epoch.append(loss_test)

  net= torch.nn.Sequential(torch.nn.Linear(1, 200))
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  opt= torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.95, weight_decay=0.005)
  sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
         #mode='min', factor=0.5, patience=100, cooldown=100,
         mode='min', factor=0.5, patience=50, cooldown=50,
         #threshold=0.0001, threshold_mode='rel',
         #threshold=-0.001, threshold_mode='rel',
         threshold=-0.01, threshold_mode='rel',
         verbose=True)

  log_lr= []
  for i_epoch,(loss_train, loss_test) in enumerate(zip(log_loss_per_epoch,log_loss_test_per_epoch)):
    if isinstance(sch,torch.optim.lr_scheduler.ReduceLROnPlateau):
      sch.step(loss_test)
    else:
      sch.step()
    log_lr.append(float(opt.param_groups[0]['lr']))

  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_lr= ax_lc.twinx()
  ax_lr.plot(range(len(log_lr)), log_lr, color='green', label='LR')
  #ax_lr.set_title('Learning rate')
  ax_lr.set_xlabel('epoch')
  ax_lr.set_ylabel('Learning rate')
  #ax_lr.set_yscale('log')
  ax_lr.legend(bbox_to_anchor=(1.0,0.8))
  fig1.tight_layout()

  plt.show()
