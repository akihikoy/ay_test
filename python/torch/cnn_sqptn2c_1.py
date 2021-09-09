#!/usr/bin/python3
#\file    cnn_sqptn2c_1.py
#\brief   Learning the square pattern 2c (classification ver) task with CNN on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.08, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image as PILImage
import os

#A_SIZE= 0.5
#A_SIZE= 0.3
A_SIZE= 0.2
DATASET_ROOT= 'data_generated/sqptn2/{}/'.format(A_SIZE)
#DATASET_ROOT= 'data_generated/sqptn2s/{}/'.format(A_SIZE)  #Smaller dataset.
#DATASET_ROOT= 'data_generated/sqptn2l/{}/'.format(A_SIZE)  #Larger dataset.

#NUM_CLASSES= 4
#def OutfeatToClass(out_feat):
  ##out_feat in [-0.5*A_SIZE, 0.5*A_SIZE]
  #if out_feat<-0.25*A_SIZE:  return 0
  #if out_feat<0.0:           return 1
  #if out_feat<0.25*A_SIZE:   return 2
  #return 3
NUM_CLASSES= 3
def OutfeatToClass(out_feat):
  logical_and= torch.logical_and if isinstance(out_feat,torch.Tensor) else (np.logical_and if isinstance(out_feat,np.ndarray) else lambda a,b:a and b)
  return (0*(out_feat<-0.25*A_SIZE)
        + 1*(logical_and(-0.25*A_SIZE<=out_feat,out_feat<=0.25*A_SIZE))
        + 2*(0.25*A_SIZE<out_feat))
def ClassToOutfeat(cls):
  return ((-0.25)*(cls==0)
        + (0.0)*(cls==1)
        + (0.25)*(cls==2)).reshape(-1,1)

'''
Generate the dataset by:
$ ./gen_sqptn2.py train [A_SIZE]
$ ./gen_sqptn2.py test [A_SIZE]
'''
class SqPtn2cDataset(torch.utils.data.Dataset):
  def __init__(self, root=DATASET_ROOT, transform=None, train=True):
    self.transform= transform
    self.image_paths= []
    self.in_feats= []
    self.out_classes= []
    self.root= root
    self.MakeIOList(train)

  def LoadValue(self, filepath):
    with open(filepath,'r') as fp:
      return float(fp.read().strip())

  def MakeIOList(self, train):
    dir_train= 'train' if train else 'test'
    imagepath_infeat_outfeat= [
        (os.path.join(self.root, dir_train, 'input', filename),
         self.LoadValue(os.path.join(self.root, dir_train, 'input', filename.replace('.jpg','.dat'))),
         self.LoadValue(os.path.join(self.root, dir_train, 'output', filename.replace('.jpg','.dat'))) )
        for filename in os.listdir(os.path.join(self.root, dir_train, 'input'))
        if filename[-4:]=='.jpg']
    self.image_paths= [image_path for image_path,in_feat,out_feat in imagepath_infeat_outfeat]
    self.in_feats= [in_feat for image_path,in_feat,out_feat in imagepath_infeat_outfeat]
    self.out_classes= [OutfeatToClass(out_feat) for image_path,in_feat,out_feat in imagepath_infeat_outfeat]

  def __getitem__(self, index):
    with open(self.image_paths[index],'rb') as f:
      img= PILImage.open(f)
      img= img.convert('RGB')
    img= img if self.transform is None else self.transform(img)
    tr_valuef= lambda v: torch.autograd.Variable(torch.tensor([v]))
    tr_valuec= lambda v: torch.autograd.Variable(torch.tensor(v))
    return img, tr_valuef(self.in_feats[index]), tr_valuec(self.out_classes[index])

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
  def __init__(self, img_shape, p_dropout=0.05):
    super(TAlexNet,self).__init__()
    self.net_img= torch.nn.Sequential(
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
    n_img_out= self.net_img(torch.FloatTensor(*((1,)+img_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img_out:',n_img_out)
    self.net_fc= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_img_out+1, 4096),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(4096, 4096),
          torch.nn.ReLU(inplace=True),
          torch.nn.Linear(4096, NUM_CLASSES)
          )

  def forward(self, x, y):
    x= self.net_img(x)
    x= x.view(x.size(0), -1)
    x= torch.cat((x,y),1)
    return self.net_fc(x)

class TCNN1(torch.nn.Module):
  def __init__(self, img_shape, p_dropout=0.05):
    super(TCNN1,self).__init__()
    self.net_img= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img_out= self.net_img(torch.FloatTensor(*((1,)+img_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img_out:',n_img_out)
    #self.net_fc1= torch.nn.Sequential(
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(n_img_out, 4096),
          #torch.nn.ReLU(inplace=True),
          ##torch.nn.Dropout(p=p_dropout),
          ##torch.nn.Linear(4096, 4096),
          #)
    #self.net_fc2= torch.nn.Sequential(
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(4096+1, 4096),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(4096, 1),
          #)
    #self.net_fc= torch.nn.Sequential(
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(n_img_out+1, 4096),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Dropout(p=p_dropout),
          #torch.nn.Linear(4096, 4096),
          #torch.nn.ReLU(inplace=True),
          #torch.nn.Linear(4096, 1)
          #)
    self.net_fc1= torch.nn.Sequential(
          #torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(1, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          )
    self.net_fc2= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_img_out+2048, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(2048, 2048),
          torch.nn.ReLU(inplace=True),
          torch.nn.Linear(2048, NUM_CLASSES),
          )

  def forward(self, x, y):
    x= self.net_img(x)
    x= x.view(x.size(0), -1)
    y= self.net_fc1(y)
    x= torch.cat((x,y),1)
    return self.net_fc2(x)
    #x= x.view(x.size(0), -1)
    #x= torch.cat((x,y),1)
    #return self.net_fc(x)

class TCNN2(torch.nn.Module):
  def __init__(self, img_shape, p_dropout=0.05):
    super(TCNN2,self).__init__()
    self.net_img= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          )
    n_img_out= self.net_img(torch.FloatTensor(*((1,)+img_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img_out:',n_img_out)
    self.net_fc1= torch.nn.Sequential(
          torch.nn.Linear(1, 1024),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(1024, 1024),
          )
    self.net_fc2= torch.nn.Sequential(
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_img_out+1024, 1024),
          torch.nn.ReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(1024, 1024),
          torch.nn.ReLU(inplace=True),
          torch.nn.Linear(1024, NUM_CLASSES),
          )

  def forward(self, x, y):
    x= self.net_img(x)
    x= x.view(x.size(0), -1)
    y= self.net_fc1(y)
    x= torch.cat((x,y),1)
    return self.net_fc2(x)

if __name__=='__main__':
  import sys
  initial_model_file= sys.argv[1] if len(sys.argv)>1 else None

  dataset_train= SqPtn2cDataset(transform=GetDataTransforms('train'), train=True)
  dataset_test= SqPtn2cDataset(transform=GetDataTransforms('eval'), train=False)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input img type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] input feat value:',dataset_train[0][1])
  print('dataset_train[0] output class value:',dataset_train[0][2])
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input img type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] input feat value:',dataset_test[0][1])
  print('dataset_test[0] output class value:',dataset_test[0][2])
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img,in_feat,out_class= dataset_train[i_data]
    in_feat= in_feat.item()
    out_class= out_class.item()
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/in={1:.3f}\nout={2}'.format(i_data,in_feat,out_class), fontsize=10)
    ax.imshow(img.permute(1,2,0))
  fig.tight_layout()
  plt.show()
  '''

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  #net= TAlexNet(img_shape=dataset_train[0][0].shape)
  net= TCNN1(img_shape=dataset_train[0][0].shape)
  #net= TCNN2(img_shape=dataset_train[0][0].shape)

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
  ##opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
  opt= torch.optim.SGD(net.parameters(), lr=0.0002, momentum=0.9, weight_decay=5e-4)
  ##opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-8)
  ##opt= torch.optim.Adagrad(net.parameters())
  ##opt= torch.optim.RMSprop(net.parameters())
  loss= torch.nn.CrossEntropyLoss()

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 20
  N_epoch= 500

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
  best_net_acc= None
  best_net_mse= None
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
    for i_step, (batch_imgs, batch_infeats, batch_outclasses) in enumerate(loader_train):
      #torch.autograd.Variable()
      b_imgs= batch_imgs  # no serialization.
      b_infeats= batch_infeats
      b_outclasses= batch_outclasses
      b_imgs,b_infeats,b_outclasses= b_imgs.to(device),b_infeats.to(device),b_outclasses.to(device)

      pred= net(b_imgs,b_infeats)
      err= loss(pred, b_outclasses)  # must be (1. nn output, 2. target)

      opt.zero_grad()  # clear gradients for next train
      err.backward()
      opt.step()
      acc= (pred.max(1).indices==b_outclasses).sum().item()/len(b_outclasses)
      log_loss_per_epoch[-1]+= err.item()/len(loader_train)
      log_acc_per_epoch[-1]+= acc/len(loader_train)
      #print(i_epoch,i_step,err)
    log_train_time[-1]= time.time()-log_train_time[-1]

    #Test the network with the test data.
    log_loss_test_per_epoch.append(0.0)
    log_acc_test_per_epoch.append(0.0)
    log_test_time.append(time.time())
    mse= 0.0  #MSE to compare with the regression.
    net.eval()  # evaluation mode; disabling dropout.
    with torch.no_grad():  # suppress calculating gradients.
      for i_step, (batch_imgs, batch_infeats, batch_outclasses) in enumerate(loader_test):
        b_imgs= batch_imgs  # no serialization.
        b_infeats= batch_infeats
        b_outclasses= batch_outclasses
        b_imgs,b_infeats,b_outclasses= b_imgs.to(device),b_infeats.to(device),b_outclasses.to(device)
        pred= net(b_imgs,b_infeats)
        err= loss(pred, b_outclasses)  # must be (1. nn output, 2. target)
        acc= (pred.max(1).indices==b_outclasses).sum().item()/len(b_outclasses)
        log_loss_test_per_epoch[-1]+= err.item()/len(loader_test)
        log_acc_test_per_epoch[-1]+= acc/len(loader_test)
        mse+= torch.mean((ClassToOutfeat(pred.max(1).indices)-ClassToOutfeat(b_outclasses))**2).item()/len(loader_test)
        #print(i_epoch,i_step,err)
    log_test_time[-1]= time.time()-log_test_time[-1]
    if best_net_state is None or log_acc_test_per_epoch[-1]>best_net_acc:
      best_net_state= copy.deepcopy(net.state_dict())
      best_net_loss= log_loss_test_per_epoch[-1]
      best_net_acc= log_acc_test_per_epoch[-1]
      best_net_mse= mse
    print(i_epoch,log_loss_per_epoch[-1],log_loss_test_per_epoch[-1],log_acc_per_epoch[-1],log_acc_test_per_epoch[-1],mse)
  print('training time:',np.sum(log_train_time))
  print('testing time:',np.sum(log_test_time))
  print('best loss:',best_net_loss)
  print('best acc:',best_net_acc)
  print('best mse:',best_net_mse)

  #Recall the best net parameters:
  net.load_state_dict(best_net_state)

  #Save the model parameters into a file.
  #To load it: net.load_state_dict(torch.load(FILEPATH))
  torch.save(net.state_dict(), 'model_learned/cnn_sqptn2c_1-{}.pt'.format(A_SIZE))

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
    img,in_feat,out_class= dataset_test[i_data]
    pred= net(img.view((1,)+img.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device)).max(1).indices.item()
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    in_feat= in_feat.item()
    out_class= out_class.item()
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}/in={1:.3f}\nout={2}\n/pred={3}'.format(i_data,in_feat,out_class,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  plt.show()
  #'''
