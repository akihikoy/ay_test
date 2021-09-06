#!/usr/bin/python3
#\file    reg_linear1.py
#\brief   PyTorch NN for a linear regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.06, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def Func(x):
  #return 2.0*x + 0.5
  return 20.0*x + 5.0
  #return 200.0*x + 50.0

def GenerateSample(xmin, xmax, N_sample, f, noise=1.0e-10):
  data_x= np.random.uniform(xmin,xmax,size=(N_sample,1))
  data_y= f(data_x) + np.random.normal(scale=noise,size=(N_sample,1))
  return data_x,data_y

class TLinear1(torch.nn.Module):
  def __init__(self):
    super(TLinear1,self).__init__()
    self.lin1= torch.nn.Linear(in_features=1, out_features=1, bias=True)
  def forward(self, x):
    x= self.lin1(x)
    return x

def GetLinearWeightByLeastSq(data_x, data_f, f_reg=1e-8):
  X= np.array(data_x)
  V= np.array(data_f)
  Theta= np.hstack((X,np.ones((len(X),1))))
  w= np.linalg.inv(Theta.T.dot(Theta)+f_reg*np.eye(Theta.shape[1])).dot(Theta.T).dot(V)
  return w

if __name__=='__main__':
  xmin,xmax= -5.0, 5.0
  N_sample= 200
  #NOTE: Adjust the sample size and noise level.
  data_x,data_y= GenerateSample(xmin, xmax, N_sample, Func, noise=2.5)

  #Convert data to torch variables.
  data_x= torch.autograd.Variable(torch.from_numpy(data_x).float())
  data_y= torch.autograd.Variable(torch.from_numpy(data_y).float())

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  net= TLinear1()

  #NOTE: Switch the device.
  device= 'cpu'
  #device= 'cuda'  # recommended to check by torch.cuda.is_available()
  net= net.to(device)

  print(net)

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  #opt= torch.optim.Adam(net.parameters(), lr=0.1)
  opt= torch.optim.SGD(net.parameters(), lr=0.02)
  #opt= torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
  #opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-3)
  ##opt= torch.optim.Adagrad(net.parameters())
  ##opt= torch.optim.RMSprop(net.parameters())
  loss= torch.nn.MSELoss()

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 50
  N_epoch= 50

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
    print(i_epoch,log_loss_per_epoch[-1])
  print('training time:',time.time()-t_0)

  print('weights:',net.lin1.state_dict()['weight'].item(),net.lin1.state_dict()['bias'].item())
  print('least_sq:',GetLinearWeightByLeastSq(data_x.numpy(),data_y.numpy()).ravel())


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

  ax_lc.set_title('Learning curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_pred.set_title('Prediction')
  ax_pred.set_xlabel('x')
  ax_pred.set_ylabel('y')
  ax_pred.legend()
  plt.show()
