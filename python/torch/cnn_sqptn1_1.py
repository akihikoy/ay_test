#!/usr/bin/python3
#\file    cnn_sqptn1_1.py
#\brief   Learning the square pattern task with CNN on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.19, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image as PILImage
import os

'''
Generate the dataset by:
$ ./gen_sqptn1.py
$ nano gen_sqptn1.py    #switch the train/test constant.
$ ./gen_sqptn1.py
'''
class SqPtn1Dataset(torch.utils.data.Dataset):
  def __init__(self, root='data_generated/sqptn1/', transform=None, train=True):
    self.transform= transform
    self.image_paths= []
    self.labels= []
    self.root= root
    self.MakePathLabelList(train)

  def LoadLabel(self, filepath):
    with open(filepath,'r') as fp:
      return float(fp.read().strip())

  def MakePathLabelList(self, train):
    dir_train= 'train' if train else 'test'
    image_paths__labels= [
        (os.path.join(self.root, dir_train, 'images', filename),
         self.LoadLabel(os.path.join(self.root, dir_train, 'labels', filename+'.dat')) )
        for filename in os.listdir(os.path.join(self.root, dir_train, 'images')) ]
    self.image_paths= [image_path for image_path,label in image_paths__labels]
    self.labels= [label for image_path,label in image_paths__labels]

  def __getitem__(self, index):
    with open(self.image_paths[index],'rb') as f:
      img= PILImage.open(f)
      img= img.convert('RGB')
    img= img if self.transform is None else self.transform(img)
    tr_value= lambda v: torch.autograd.Variable(torch.tensor([v]))
    return img, tr_value(self.labels[index])

  def __len__(self):
    return len(self.image_paths)

def GetDataTransforms(mode):
  if mode=='train':
    return torchvision.transforms.Compose([
        #torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
  if mode=='eval':
    return torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
  if mode=='none':
    return torchvision.transforms.Compose([
        #torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
      ])

class TAlexNet(torch.nn.Module):
  def __init__(self, img_shape, p_dropout=0.02):
    super(TAlexNet,self).__init__()
    self.net_features= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_feat_out= self.net_features(torch.FloatTensor(*((1,)+img_shape))).view(1,-1).shape[1]
    self.net_estimator= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_feat_out, 4096),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(4096, 4096),
          torch.nn.ReLU(inplace=True),
          torch.nn.Linear(4096, 1)
          )

  def forward(self, x):
    x= self.net_features(x)
    x= x.view(x.size(0), -1)
    return self.net_estimator(x)

if __name__=='__main__':
  import sys
  initial_model_file= sys.argv[1] if len(sys.argv)>1 else None

  dataset_train= SqPtn1Dataset(transform=GetDataTransforms('train'), train=True)
  dataset_test= SqPtn1Dataset(transform=GetDataTransforms('eval'), train=False)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] label value:',dataset_train[0][1])
  #print('dataset_train[0][0] value:',dataset_train[0][0][:,100:110,100:110])
  print('dataset_train[0][0] value range:',torch.min(dataset_train[0][0]),torch.max(dataset_train[0][0]))
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] label value:',dataset_test[0][1])
  print('dataset_test[0][0] value range:',torch.min(dataset_test[0][0]),torch.max(dataset_test[0][0]))
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img,label= dataset_train[i_data]
    label= label.item()
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/l={1:.4f}'.format(i_data,label), fontsize=10)
    ax.imshow(img.permute(1,2,0))
  fig.tight_layout()
  plt.show()
  '''

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  net= TAlexNet(img_shape=dataset_train[0][0].shape)

  #NOTE: Switch the device.
  #device= 'cpu'
  device= 'cuda'
  if device=='cuda' and not torch.cuda.is_available():
    device= 'cpu'
    print('device is modified to cpu since cuda is not available.')
  net= net.to(device)

  if initial_model_file is not None:
    net.load_state_dict(torch.load(initial_model_file, map_location=device))

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
  loss= torch.nn.MSELoss()

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 10
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

  best_net_state= None
  best_net_loss= None
  log_train_time= []
  log_test_time= []
  log_loss_per_epoch= []
  log_loss_test_per_epoch= []
  for i_epoch in range(N_epoch):
    log_loss_per_epoch.append(0.0)
    log_train_time.append(time.time())
    net.train()  # training mode; using dropout.
    for i_step, (batch_imgs, batch_labels) in enumerate(loader_train):
      #torch.autograd.Variable()
      b_imgs= batch_imgs  # no serialization.
      b_labels= batch_labels
      b_imgs,b_labels= b_imgs.to(device),b_labels.to(device)

      pred= net(b_imgs)
      err= loss(pred, b_labels)  # must be (1. nn output, 2. target)

      opt.zero_grad()  # clear gradients for next train
      err.backward()
      opt.step()
      log_loss_per_epoch[-1]+= err.item()/len(loader_train)
      #print(i_epoch,i_step,err)
    log_train_time[-1]= time.time()-log_train_time[-1]

    #Test the network with the test data.
    log_loss_test_per_epoch.append(0.0)
    log_test_time.append(time.time())
    net.eval()  # evaluation mode; disabling dropout.
    with torch.no_grad():  # suppress calculating gradients.
      for i_step, (batch_imgs, batch_labels) in enumerate(loader_test):
        b_imgs= batch_imgs  # no serialization.
        b_labels= batch_labels
        b_imgs,b_labels= b_imgs.to(device),b_labels.to(device)
        pred= net(b_imgs)
        err= loss(pred, b_labels)  # must be (1. nn output, 2. target)
        log_loss_test_per_epoch[-1]+= err.item()/len(loader_test)
        #print(i_epoch,i_step,err)
    log_test_time[-1]= time.time()-log_test_time[-1]
    if best_net_state is None or log_loss_test_per_epoch[-1]<best_net_loss:
      best_net_state= copy.deepcopy(net.state_dict())
      best_net_loss= log_loss_test_per_epoch[-1]
    print(i_epoch,log_loss_per_epoch[-1],log_loss_test_per_epoch[-1])
  print('training time:',np.sum(log_train_time))
  print('testing time:',np.sum(log_test_time))
  print('best loss:',best_net_loss)

  #Recall the best net parameters:
  net.load_state_dict(best_net_state)

  #Save the model parameters into a file.
  #To load it: net.load_state_dict(torch.load(FILEPATH))
  torch.save(net.state_dict(), 'model_learned/cnn_sqptn1_1.pt')

  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  fig1.tight_layout()

  net.eval()  # evaluation mode; disabling dropout.
  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_test)))
    img,label= dataset_test[i_data]
    label= label.item()
    pred= net(img.view((1,)+img.shape).to(device)).data.cpu().item()
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}\n/l={1:.4f}\n/pred={2:.4f}'.format(i_data,label,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  plt.show()
  #'''
