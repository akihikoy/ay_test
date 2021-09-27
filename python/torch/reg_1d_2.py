#!/usr/bin/python3
#\file    reg_1d_2.py
#\brief   Comparison of loss functions in the reg_1d_1 task.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.27, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from reg_1d_1 import *

def RunTest(test_name, data_x, data_y, net, device, opt, loss, N_batch, N_epoch):
  print('Running test {}...'.format(test_name))
  print(net)

  torch_dataset= torch.utils.data.TensorDataset(data_x, data_y)
  loader= torch.utils.data.DataLoader(
          dataset=torch_dataset,
          batch_size=N_batch,
          shuffle=True,
          num_workers=2)

  t_0= time.time()
  net.train()  # training mode; using dropout.
  log_loss_per_epoch= []
  for i_epoch in range(N_epoch):
    log_loss_per_epoch.append(0.0)
    for i_step, (batch_x, batch_y) in enumerate(loader):
      b_x= torch.autograd.Variable(batch_x)
      b_y= torch.autograd.Variable(batch_y)
      b_x,b_y= b_x.to(device),b_y.to(device)

      pred= net(b_x)
      err= loss(pred, b_y)  # must be (1. nn output, 2. target)

      opt.zero_grad()  # clear gradients for next train
      err.backward()
      opt.step()
      log_loss_per_epoch[-1]+= err.item()/len(loader)
      #print(i_epoch,i_step,err)
    if i_epoch%20==0:  print(i_epoch,log_loss_per_epoch[-1])
  print(test_name,': training time:',time.time()-t_0)


  #print(data_x,data_y)
  fig, (ax_lc,ax_pred) = plt.subplots(1,2,figsize=(10,5))
  true_x= np.linspace(xmin,xmax,1000).reshape((-1,1))
  ax_pred.plot(true_x, Func(true_x), color='green', linewidth=1, label='true_func')
  ax_pred.scatter(data_x, data_y, 1, color='blue', label='data')

  net.eval()  # evaluation mode; disabling dropout.
  true_x_var= torch.from_numpy(true_x).float()
  true_x_var= true_x_var.to(device)
  ax_pred.plot(true_x, net(true_x_var).data.cpu(), color='red', linewidth=2, label='nn_reg')

  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss')

  ax_lc.set_title('{}: Learning curve'.format(test_name))
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_pred.set_title('{}: Prediction'.format(test_name))
  ax_pred.set_xlabel('x')
  ax_pred.set_ylabel('y')
  ax_pred.legend()

if __name__=='__main__':
  xmin,xmax= -5.0, 5.0
  N_sample= 50
  N_outliers= N_sample//4
  #NOTE: Adjust the sample size and noise level.
  data_x,data_y= GenerateSample(xmin, xmax, N_sample, Func, noise=0.5)
  data_y[np.random.permutation(range(N_sample))[:N_outliers]]+= np.random.normal(scale=5.0,size=(N_outliers,1))

  #Convert data to torch variables.
  data_x= torch.autograd.Variable(torch.from_numpy(data_x).float())
  data_y= torch.autograd.Variable(torch.from_numpy(data_y).float())

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  #net_class= TRegNN1
  #net_class= DefRegNN2
  #net_class= DefRegNN3
  #net_class= DefRegNN4
  net_class= DefRegNN5

  #NOTE: Switch the device.
  device= 'cpu'
  #device= 'cuda'  # recommended to check by torch.cuda.is_available()

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  #opt= torch.optim.SGD(net.parameters(), lr=0.004)
  #opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  #opt= torch.optim.Adadelta(net.parameters(), rho=0.9, eps=1e-8)
  #opt= torch.optim.Adagrad(net.parameters())
  #opt= torch.optim.RMSprop(net.parameters())

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 50
  N_epoch= 100

  net= net_class()
  net= net.to(device)
  opt= torch.optim.Adam(net.parameters(), lr=0.01)
  loss= torch.nn.MSELoss()
  RunTest('MSELoss', data_x, data_y, net, device, opt, loss, N_batch, N_epoch)

  net= net_class()
  net= net.to(device)
  opt= torch.optim.Adam(net.parameters(), lr=0.01)
  loss= torch.nn.L1Loss()
  RunTest('L1Loss', data_x, data_y, net, device, opt, loss, N_batch, N_epoch)

  net= net_class()
  net= net.to(device)
  opt= torch.optim.Adam(net.parameters(), lr=0.01)
  #opt= torch.optim.SGD(net.parameters(), lr=0.2)
  loss= torch.nn.HuberLoss(reduction='mean', delta=0.5)
  RunTest('HuberLoss', data_x, data_y, net, device, opt, loss, N_batch, N_epoch)

  plt.show()
