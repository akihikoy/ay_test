#!/usr/bin/python3
#\file    multiple_losses_1.py
#\brief   Testing learning under multiple losses;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.05, 2021
#ref. https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

FUNC1_KIND=3
def Func1(x):
  #NOTE: Switch the function to be learned.
  global FUNC1_KIND
  if FUNC1_KIND==0:  return x
  if FUNC1_KIND==1:  return 0.5*x**2
  if FUNC1_KIND==2:  return 0.1*x**3
  if FUNC1_KIND==3:  return 0.1*x**3-1.0*x

FUNC2_KIND=3
def Func2(x):
  #NOTE: Switch the function to be learned.
  global FUNC2_KIND
  if FUNC2_KIND==0:  return x
  if FUNC2_KIND==1:  return (x[:,0]**2).reshape((-1,1))
  if FUNC2_KIND==2:  return (3.0-x[:,0]).reshape((-1,1))
  if FUNC2_KIND==3:  return (x[:,0]+3.0*np.sin(x[:,0])).reshape((-1,1))
  if FUNC2_KIND==4:  return np.where(x[:,0]**2<1.0, 3.0, 0.0).reshape((-1,1))

def GenerateSample(xmin, xmax, N_sample, noise=1.0e-10):
  data_x1= np.random.uniform(xmin,xmax,size=(N_sample,1))
  data_x2= Func1(data_x1) + np.random.normal(scale=noise,size=(N_sample,1))
  data_y= Func2(data_x2) + np.random.normal(scale=noise,size=(N_sample,1))
  return data_x1,data_x2,data_y

class TFCN1(torch.nn.Module):
  def __init__(self, p_dropout=0.02):
    super(TFCN1,self).__init__()
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 100),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(100, 100),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(100, 1),
          )
    self.net_fc2= torch.nn.Sequential(
          torch.nn.Linear(1, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 200),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(200, 1),
          )
  def forward(self, x):
    x2= self.net_fc1(x)
    y= self.net_fc2(x2)
    return x2,y


if __name__=='__main__':
  xmin,xmax= -5.0, 5.0
  N_sample= 2000
  #NOTE: Adjust the sample size and noise level.
  data_x1,data_x2,data_y= GenerateSample(xmin, xmax, N_sample, noise=0.1)

  #Convert data to torch variables.
  data_x1= torch.autograd.Variable(torch.from_numpy(data_x1).float())
  data_x2= torch.autograd.Variable(torch.from_numpy(data_x2).float())
  data_y= torch.autograd.Variable(torch.from_numpy(data_y).float())

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  net= TFCN1()

  #NOTE: Switch the device.
  device= 'cpu'
  #device= 'cuda'  # recommended to check by torch.cuda.is_available()
  net= net.to(device)

  print(net)

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  opt= torch.optim.Adam(net.parameters(), lr=0.01)
  #opt= torch.optim.SGD(net.parameters(), lr=0.004)
  #opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  #opt= torch.optim.Adadelta(net.parameters(), rho=0.9, eps=1e-8)
  #opt= torch.optim.Adagrad(net.parameters())
  #opt= torch.optim.RMSprop(net.parameters())
  loss_x2= torch.nn.HuberLoss(reduction='mean', delta=0.2)
  loss_y= torch.nn.HuberLoss(reduction='mean', delta=1.0)
  #loss_x2= torch.nn.MSELoss()
  #loss_y= torch.nn.MSELoss()

  #update_method= 'both'
  #update_method= 'only_y'
  update_method= 'delayed'
  i_delay= 20

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 50
  N_epoch= 100

  torch_dataset= torch.utils.data.TensorDataset(data_x1, data_x2, data_y)
  loader= torch.utils.data.DataLoader(
          dataset=torch_dataset,
          batch_size=N_batch,
          shuffle=True,
          num_workers=2)

  t_0= time.time()
  net.train()  # training mode; using dropout.
  log_loss_x2_per_epoch= []
  log_loss_y_per_epoch= []
  for i_epoch in range(N_epoch):
    log_loss_x2_per_epoch.append(0.0)
    log_loss_y_per_epoch.append(0.0)
    for i_step, (batch_x1, batch_x2, batch_y) in enumerate(loader):
      b_x1= torch.autograd.Variable(batch_x1)
      b_x2= torch.autograd.Variable(batch_x2)
      b_y= torch.autograd.Variable(batch_y)
      b_x1,b_x2,b_y= b_x1.to(device),b_x2.to(device),b_y.to(device)

      if update_method=='both':
        opt.zero_grad()
        pred_x2,pred_y= net(b_x1)
        err_x2= 0.2*loss_x2(pred_x2, b_x2)  # must be (1. nn output, 2. target)
        err_y= loss_y(pred_y, b_y)  # must be (1. nn output, 2. target)
        err_x2.backward(retain_graph=True)
        err_y.backward()
        opt.step()
        log_loss_x2_per_epoch[-1]+= err_x2.item()/len(loader)
        log_loss_y_per_epoch[-1]+= err_y.item()/len(loader)
      elif update_method=='only_y':
        opt.zero_grad()
        pred_x2,pred_y= net(b_x1)
        err_y= loss_y(pred_y, b_y)  # must be (1. nn output, 2. target)
        err_y.backward()
        opt.step()
        log_loss_y_per_epoch[-1]+= err_y.item()/len(loader)
      elif update_method=='delayed':
        opt.zero_grad()
        pred_x2,pred_y= net(b_x1)
        err_x2= 0.2*loss_x2(pred_x2, b_x2)  # must be (1. nn output, 2. target)
        if i_epoch>i_delay:  err_y= loss_y(pred_y, b_y)  # must be (1. nn output, 2. target)
        err_x2.backward(retain_graph=True)
        if i_epoch>i_delay:  err_y.backward()
        opt.step()
        log_loss_x2_per_epoch[-1]+= err_x2.item()/len(loader)
        if i_epoch>i_delay:  log_loss_y_per_epoch[-1]+= err_y.item()/len(loader)
    print(i_epoch,log_loss_x2_per_epoch[-1],log_loss_y_per_epoch[-1])
  print('training time:',time.time()-t_0)


  #print(data_x,data_y)
  fig,(ax_lc,ax_pred_x2,ax_pred_x1_y,ax_pred_x2_y)= plt.subplots(1,4,figsize=(20,5))
  true_x1= np.linspace(xmin,xmax,1000).reshape((-1,1))
  true_x2= Func1(true_x1)
  true_y= Func2(true_x2)

  ax_pred_x2.plot(true_x1, true_x2, color='green', linewidth=1, label='true_x2')
  ax_pred_x2.scatter(data_x1, data_x2, 1, color='blue', label='data_x2')
  ax_pred_x1_y.plot(true_x1, true_y, color='green', linewidth=1, label='true_y(x1)')
  ax_pred_x1_y.scatter(data_x1, data_y, 1, color='blue', label='data_y(x1)')
  ax_pred_x2_y.plot(true_x2, true_y, color='green', linewidth=1, label='true_y(x2)')
  ax_pred_x2_y.scatter(data_x2, data_y, 1, color='blue', label='data_y(x2)')

  net.eval()  # evaluation mode; disabling dropout.
  pred_x2,pred_y= net(torch.from_numpy(true_x1).float().to(device))
  pred_x2,pred_y= pred_x2.data.cpu(),pred_y.data.cpu()
  ax_pred_x2.plot(true_x1, pred_x2, color='red', linewidth=2, label='pred_x2')
  ax_pred_x1_y.plot(true_x1, pred_y, color='red', linewidth=2, label='pred_y(x1)')
  ax_pred_x2_y.plot(pred_x2, pred_y, color='red', linewidth=2, label='pred_y(x2)')

  ax_lc.plot(range(len(log_loss_x2_per_epoch)), log_loss_x2_per_epoch, color='blue', label='loss_x2')
  ax_lc.plot(range(len(log_loss_y_per_epoch)), log_loss_y_per_epoch, color='green', label='loss_y')

  ax_lc.set_title('Learning curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_pred_x2.set_title('Prediction(x2(x1))')
  ax_pred_x2.set_xlabel('x1')
  ax_pred_x2.set_ylabel('x2')
  ax_pred_x2.legend()
  ax_pred_x1_y.set_title('Prediction(y(x1)')
  ax_pred_x1_y.set_xlabel('x1')
  ax_pred_x1_y.set_ylabel('y')
  ax_pred_x1_y.legend()
  ax_pred_x2_y.set_title('Prediction(y(x2)')
  ax_pred_x2_y.set_xlabel('x2')
  ax_pred_x2_y.set_ylabel('y')
  ax_pred_x2_y.legend()
  plt.show()
