#!/usr/bin/python3
#\file    cnn_sqptn2_1.py
#\brief   Learning the square pattern 2 task with CNN on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.03, 2021
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
OUTFEAT_SCALE= 1.0/A_SIZE

'''
Generate the dataset by:
$ ./gen_sqptn2.py train [A_SIZE]
$ ./gen_sqptn2.py test [A_SIZE]
'''
class SqPtn2Dataset(torch.utils.data.Dataset):
  def __init__(self, root=DATASET_ROOT, transform=None, train=True):
    self.transform= transform
    self.image_paths= []
    self.in_feats= []
    self.out_feats= []
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
    self.out_feats= [out_feat*OUTFEAT_SCALE for image_path,in_feat,out_feat in imagepath_infeat_outfeat]

  def __getitem__(self, index):
    with open(self.image_paths[index],'rb') as f:
      img= PILImage.open(f)
      img= img.convert('RGB')
    img= img if self.transform is None else self.transform(img)
    tr_value= lambda v: torch.autograd.Variable(torch.tensor([v]))
    return img, tr_value(self.in_feats[index]), tr_value(self.out_feats[index])

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
          torch.nn.Linear(4096, 1)
          )

  def forward(self, x, y):
    x= self.net_img(x)
    x= x.view(x.size(0), -1)
    x= torch.cat((x,y),1)
    return self.net_fc(x)

class TCNN1(torch.nn.Module):
  def __init__(self, img_shape, p_dropout=0.02):
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
          torch.nn.Linear(2048, 1),
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
  def __init__(self, img_shape, p_dropout=0.02):
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
          torch.nn.Linear(1024, 1),
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

  dataset_train= SqPtn2Dataset(transform=GetDataTransforms('train'), train=True)
  dataset_test= SqPtn2Dataset(transform=GetDataTransforms('eval'), train=False)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input img type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] input feat value:',dataset_train[0][1])
  print('dataset_train[0] output feat value:',dataset_train[0][2])
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input img type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] input feat value:',dataset_test[0][1])
  print('dataset_test[0] output feat value:',dataset_test[0][2])
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img,in_feat,out_feat= dataset_train[i_data]
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/in={1:.3f}\nout={2:.3f}'.format(i_data,in_feat,out_feat), fontsize=10)
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
  opt= torch.optim.Adam(net.parameters(), lr=0.001)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.004)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  #opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  #opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-8)
  ##opt= torch.optim.Adagrad(net.parameters())
  ##opt= torch.optim.RMSprop(net.parameters())
  loss= torch.nn.MSELoss()

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 20
  N_epoch= 100

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
    for i_step, (batch_imgs, batch_infeats, batch_outfeats) in enumerate(loader_train):
      #torch.autograd.Variable()
      b_imgs= batch_imgs  # no serialization.
      b_infeats= batch_infeats
      b_outfeats= batch_outfeats
      b_imgs,b_infeats,b_outfeats= b_imgs.to(device),b_infeats.to(device),b_outfeats.to(device)

      pred= net(b_imgs,b_infeats)
      err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)

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
      for i_step, (batch_imgs, batch_infeats, batch_outfeats) in enumerate(loader_test):
        b_imgs= batch_imgs  # no serialization.
        b_infeats= batch_infeats
        b_outfeats= batch_outfeats
        b_imgs,b_infeats,b_outfeats= b_imgs.to(device),b_infeats.to(device),b_outfeats.to(device)
        pred= net(b_imgs,b_infeats)
        err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)
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
  torch.save(net.state_dict(), 'model_learned/cnn_sqptn2_1-{}.pt'.format(A_SIZE))

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
    img,in_feat,out_feat= dataset_test[i_data]
    pred= net(img.view((1,)+img.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device)).data.cpu().item()/OUTFEAT_SCALE
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}/in={1:.3f}\nout={2:.3f}\n/pred={3:.3f}'.format(i_data,in_feat,out_feat,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  plt.show()
  #'''
