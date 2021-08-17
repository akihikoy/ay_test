#!/usr/bin/python3
#\file    mlp_cifar10_1.py
#\brief   Learning CIFAR-10 with MLP on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.17, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time

def DefMLP1(dim_in, dim_out, p=0.2):
  return torch.nn.Sequential(
          torch.nn.Linear(dim_in, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, dim_out),
          )

def DefMLP2(dim_in, dim_out, p=0.2):
  return torch.nn.Sequential(
          torch.nn.Linear(dim_in, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, 600),
          torch.nn.ReLU(),
          torch.nn.Dropout(p),
          torch.nn.Linear(600, dim_out),
          torch.nn.Softmax(dim=-1),
          )

if __name__=='__main__':
  dataset_train= torchvision.datasets.CIFAR10(
                    root='./data_downloaded/',
                    train=True,
                    transform=torchvision.transforms.ToTensor(),
                    download=True)
  dataset_test= torchvision.datasets.CIFAR10(
                    root='./data_downloaded/',
                    train=False,
                    transform=torchvision.transforms.ToTensor(),
                    download=True)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] label value:',dataset_train[0][1],dataset_train.classes[dataset_train[0][1]])
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] label value:',dataset_test[0][1],dataset_test.classes[dataset_test[0][1]])
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img,label= dataset_train[i_data]
    #print(i_data,type(img),img.shape)
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/l={1}'.format(i_data,dataset_train.classes[label]), fontsize=10)
    ax.imshow(img.permute(1,2,0))
  fig.tight_layout()
  plt.show()
  '''

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  net= DefMLP1(dim_in=dataset_train[0][0].numel(), dim_out=len(dataset_train.classes))
  #net= DefMLP2(dim_in=dataset_train[0][0].numel(), dim_out=len(dataset_train.classes))

  #NOTE: Switch the device.
  #device= 'cpu'
  device= 'cuda'  # recommended to check by torch.cuda.is_available()
  net= net.to(device)

  print(net)

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  #opt= torch.optim.Adam(net.parameters(), lr=0.001)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.004)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  #opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-8)
  ##opt= torch.optim.Adagrad(net.parameters())
  ##opt= torch.optim.RMSprop(net.parameters())
  loss= torch.nn.CrossEntropyLoss()

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 64
  N_epoch= 20

  loader_train= torch.utils.data.DataLoader(
                  dataset=dataset_train,
                  batch_size=N_batch,
                  shuffle=True,
                  num_workers=2)
  loader_test= torch.utils.data.DataLoader(
                  dataset=dataset_test,
                  batch_size=N_batch,
                  shuffle=False,
                  num_workers=2)

  log_train_time= []
  log_test_time= []
  log_loss_per_epoch= []
  log_acc_per_epoch= []
  log_loss_test_per_epoch= []
  log_acc_test_per_epoch= []
  for i_epoch in range(N_epoch):
    log_loss_per_epoch.append(0.0)
    log_acc_per_epoch.append(0.0)
    log_train_time.append(time.time())
    net.train()  # training mode; using dropout.
    for i_step, (batch_imgs, batch_labels) in enumerate(loader_train):
      #torch.autograd.Variable()
      b_imgs= batch_imgs.view(-1, batch_imgs[0].numel())  # serialize the images.
      b_labels= batch_labels
      b_imgs,b_labels= b_imgs.to(device),b_labels.to(device)

      pred= net(b_imgs)
      err= loss(pred, b_labels)  # must be (1. nn output, 2. target)

      opt.zero_grad()  # clear gradients for next train
      err.backward()
      opt.step()
      acc= (pred.max(1).indices==b_labels).sum().item()/len(b_labels)
      log_loss_per_epoch[-1]+= err.item()/len(loader_train)
      log_acc_per_epoch[-1]+= acc/len(loader_train)
      #print(i_epoch,i_step,err,acc)
    log_train_time[-1]= time.time()-log_train_time[-1]

    #Test the network with the test data.
    log_loss_test_per_epoch.append(0.0)
    log_acc_test_per_epoch.append(0.0)
    log_test_time.append(time.time())
    net.eval()  # evaluation mode; disabling dropout.
    with torch.no_grad():  # suppress calculating gradients.
      for i_step, (batch_imgs, batch_labels) in enumerate(loader_test):
        b_imgs= batch_imgs.view(-1, batch_imgs[0].numel())  # serialize the images.
        b_labels= batch_labels
        b_imgs,b_labels= b_imgs.to(device),b_labels.to(device)
        pred= net(b_imgs)
        err= loss(pred, b_labels)  # must be (1. nn output, 2. target)
        acc= (pred.max(1).indices==b_labels).sum().item()/len(b_labels)
        log_loss_test_per_epoch[-1]+= err.item()/len(loader_test)
        log_acc_test_per_epoch[-1]+= acc/len(loader_test)
        #print(i_epoch,i_step,err,acc)
    log_test_time[-1]= time.time()-log_test_time[-1]
    print(i_epoch,log_loss_per_epoch[-1],log_loss_test_per_epoch[-1],log_acc_per_epoch[-1],log_acc_test_per_epoch[-1])
  print('training time:',np.sum(log_train_time))
  print('testing time:',np.sum(log_test_time))

  fig1= plt.figure(figsize=(10,5))
  ax_lc= fig1.add_subplot(1,2,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  ax_acc= fig1.add_subplot(1,2,2)
  ax_acc.plot(range(len(log_acc_per_epoch)), log_acc_per_epoch, color='blue', label='acc_train')
  ax_acc.plot(range(len(log_acc_test_per_epoch)), log_acc_test_per_epoch, color='red', label='acc_test')
  ax_acc.set_title('Accuracy curve')
  ax_acc.set_xlabel('epoch')
  ax_acc.set_ylabel('accuracy')
  ax_acc.legend()
  fig1.tight_layout()

  net.eval()  # evaluation mode; disabling dropout.
  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_test)))
    img,label= dataset_test[i_data]
    pred= net(img.view(img.numel()).to(device)).max(0).indices.item()
    #print(i_data,type(img),img.shape)
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}\n/l={1}\n/pred={2}'.format(i_data,dataset_test.classes[label],dataset_test.classes[pred]), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  plt.show()

