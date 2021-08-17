#!/usr/bin/python3
#\file    reg_2d_1.py
#\brief   PyTorch NN for 2d regression.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.16, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import time

def Func(x):
  #return (x[:,0]+2.0*x[:,1]).reshape((-1,1))
  #return (x[:,0]*x[:,1]).reshape((-1,1))
  #return (-2.0*x[:,0]**2+0.5*x[:,0]*x[:,1]-x[:,1]**2).reshape((-1,1))
  return (x[:,0]*np.sin(1.0*x[:,1])).reshape((-1,1))
  #return (3.0-(x[:,0]**2+x[:,1]**2)).reshape((-1,1))
  #return (3.0-(x[:,0]**2+np.sin(3.0*x[:,1])**2)).reshape((-1,1))
  #return (np.where(np.sqrt(x[:,0]**2+x[:,1]**2) < 2.0, 5.0, 0.0)).reshape((-1,1))

def GenerateSample(xmin, xmax, N_sample, f, noise=1.0e-10):
  data_x= np.random.uniform(xmin,xmax,size=(N_sample,2))
  data_y= f(data_x) + np.random.normal(scale=noise,size=(N_sample,1))
  return data_x,data_y

class TRegNN1(torch.nn.Module):
  def __init__(self):
    super(TRegNN1,self).__init__()
    self.lin1= torch.nn.Linear(in_features=2, out_features=1, bias=True)
  def forward(self, x):
    x= self.lin1(x)
    return x

def DefRegNN2():
  return torch.nn.Sequential(
          torch.nn.Linear(2, 200),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(200, 100),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(100, 1),
          )

def DefRegNN3():
  return torch.nn.Sequential(
          torch.nn.Linear(2, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 1),
          )

def DefRegNN4():
  return torch.nn.Sequential(
          torch.nn.Linear(2, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 1),
          )

def DefRegNN5(p=0.02):
  return torch.nn.Sequential(
          torch.nn.Linear(2, 200),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(200, 200),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(200, 1),
          )

if __name__=='__main__':
  xmin,xmax= -5.0, 5.0
  #NOTE: Adjust the sample size and noise level.
  N_sample= 200
  data_x,data_y= GenerateSample(xmin, xmax, N_sample, Func, noise=0.5)

  #Convert data to torch variables.
  data_x= torch.autograd.Variable(torch.from_numpy(data_x).float())
  data_y= torch.autograd.Variable(torch.from_numpy(data_y).float())

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  #net= TRegNN1()
  #net= DefRegNN2()
  #net= DefRegNN3()
  #net= DefRegNN4()
  net= DefRegNN5()

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


  #print(data_x,data_y)
  fig= plt.figure(figsize=(10,5))
  ax_lc= fig.add_subplot(1,2,1)
  ax_pred= fig.add_subplot(1,2,2,projection='3d')

  true_x= np.mgrid[xmin:xmax:0.1, xmin:xmax:0.1]
  ax_pred.plot_wireframe(true_x[0], true_x[1], Func(true_x[:,:,:].reshape(2,-1).T).reshape(true_x.shape[1:]), color='green', linewidth=1, label='true_func')
  ax_pred.scatter(data_x[:,0], data_x[:,1], data_y[:,0], color='blue', label='data')

  net.eval()  # evaluation mode; disabling dropout.
  true_x_var= torch.from_numpy(true_x[:,:,:].reshape(2,-1).T).float()
  true_x_var= true_x_var.to(device)
  ax_pred.plot_wireframe(true_x[0], true_x[1], net(true_x_var).data.reshape(true_x.shape[1:]).cpu(), color='red', linewidth=2, label='nn_reg')

  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss')

  ax_lc.set_title('Learning curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_pred.set_title('Prediction')
  ax_pred.set_xlabel('x[0]')
  ax_pred.set_ylabel('x[1]')
  ax_pred.set_zlabel('y')
  ax_pred.legend()
  plt.show()
