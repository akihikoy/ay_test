#!/usr/bin/python3
#\file    cnn_sqptn3_2.py
#\brief   Learning the square pattern 3 task with CNN on PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.05, 2021
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image as PILImage
import os
from lr_sch_2 import ReduceLRAtCondition
import cnn_sqptn3_1


#cnn_sqptn3_1.sqptn3= 'sqptn3'
cnn_sqptn3_1.sqptn3= 'sqptn3l'  #Larger dataset.
#cnn_sqptn3_1.A_SIZE1,cnn_sqptn3_1.A_SIZE2= 0.3,0.3
cnn_sqptn3_1.A_SIZE1,cnn_sqptn3_1.A_SIZE2= 0.1,0.1

sqptn3,A_SIZE1,A_SIZE2= cnn_sqptn3_1.sqptn3,cnn_sqptn3_1.A_SIZE1,cnn_sqptn3_1.A_SIZE2
OUTFEAT_SCALE= cnn_sqptn3_1.OUTFEAT_SCALE
SqPtn3Dataset= cnn_sqptn3_1.SqPtn3Dataset
GetDataTransforms= cnn_sqptn3_1.GetDataTransforms

class TCNN2_1(torch.nn.Module):
  def __init__(self, img1_shape, img2_shape, p_dropout=0.02, n_emb=10, n_fc=300):
    super(TCNN2_1,self).__init__()
    self.net_img1= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    self.net_img2= torch.nn.Sequential(
          #torch.nn.Conv2d(in_channels, out_channels, ...)
          torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(192, 256, kernel_size=3, padding=1),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          #torch.nn.MaxPool2d(kernel_size=4, stride=4),
          )
    n_img1_out= self.net_img1(torch.FloatTensor(*((1,)+img1_shape))).view(1,-1).shape[1]
    n_img2_out= self.net_img2(torch.FloatTensor(*((1,)+img2_shape))).view(1,-1).shape[1]
    print('DEBUG:n_img1_out,n_img2_out:',n_img1_out,n_img2_out)
    self.n_emb= n_emb
    self.n_fc= n_fc
    self.net_fc2a= torch.nn.Sequential(
          torch.nn.Linear(n_img1_out, n_emb),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc2b= torch.nn.Sequential(
          torch.nn.Linear(n_img2_out, n_emb),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          )
    self.net_fc4= torch.nn.Sequential(
          torch.nn.Linear(n_emb*2, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )
    self.net_fc3= torch.nn.Sequential(
          torch.nn.Linear(n_emb*2+n_emb, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, n_fc),
          torch.nn.LeakyReLU(inplace=True),
          torch.nn.Dropout(p=p_dropout),
          torch.nn.Linear(n_fc, 1),
          )

  def forward(self, x1, x2, y):
    x1= self.net_img1(x1)
    x1= x1.view(x1.size(0), -1)
    x1= self.net_fc2a(x1)
    x2= self.net_img2(x2)
    x2= x2.view(x2.size(0), -1)
    x2= self.net_fc2b(x2)
    y= y.repeat(1, self.n_emb)
    return self.net_fc3(torch.cat((x1,x2,y),1)), self.net_fc4(torch.cat((x1,x2),1))


if __name__=='__main__':
  import sys
  initial_model_file= sys.argv[1] if len(sys.argv)>1 else None

  dataset_train= SqPtn3Dataset(transform=GetDataTransforms('train'), train=True)
  dataset_test= SqPtn3Dataset(transform=GetDataTransforms('eval'), train=False)

  #Show the dataset info.
  print('dataset_train size:',len(dataset_train))
  print('dataset_train[0] input img1 type, shape:',type(dataset_train[0][0]),dataset_train[0][0].shape)
  print('dataset_train[0] input img2 type, shape:',type(dataset_train[0][1]),dataset_train[0][1].shape)
  print('dataset_train[0] input feat value:',dataset_train[0][2])
  print('dataset_train[0] output feat value:',dataset_train[0][3])
  print('dataset_test size:',len(dataset_test))
  print('dataset_test[0] input img1 type, shape:',type(dataset_test[0][0]),dataset_test[0][0].shape)
  print('dataset_test[0] input img2 type, shape:',type(dataset_test[0][1]),dataset_test[0][1].shape)
  print('dataset_test[0] input feat value:',dataset_test[0][2])
  print('dataset_test[0] output feat value:',dataset_test[0][3])
  '''Uncomment to plot training dataset.
  fig= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img1,img2,in_feat,out_feat= dataset_train[i_data]
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    ax= fig.add_subplot(rows, cols, i+1)
    ax.set_title('train#{0}/in={1:.3f}\nout={2:.3f}'.format(i_data,in_feat,out_feat), fontsize=10)
    ax.imshow(img.permute(1,2,0))
  fig.tight_layout()
  plt.show()
  '''

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  img1_shape,img2_shape= dataset_train[0][0].shape,dataset_train[0][1].shape
  #net= TCNN2_1(img1_shape,img2_shape)
  #net= TCNN2_1(img1_shape,img2_shape,n_emb=100,n_fc=50)
  net= TCNN2_1(img1_shape,img2_shape,n_emb=300,n_fc=12)

  #NOTE: Switch the device.
  #device= 'cpu'
  device= 'cuda'
  #device= 'cuda:1'  #Second GPU.
  if device=='cuda' and not torch.cuda.is_available():
    device= 'cpu'
    print('device is modified to cpu since cuda is not available.')
  net= net.to(device)

  if initial_model_file is not None:
    net.load_state_dict(torch.load(initial_model_file, map_location=device))

  print(net)

  #NOTE: Switch the optimizer.
  #Setup an optimizer and a loss function.
  sch= None
  swa_sch= None
  ##opt= torch.optim.Adam(net.parameters(), lr=0.001)
  ###opt= torch.optim.SGD(net.parameters(), lr=0.004)
  ###opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95)
  ##opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=5e-4)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.01)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #opt= torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.95, weight_decay=5e-4)
  ##opt= torch.optim.Adadelta(net.parameters(), rho=0.95, eps=1e-8)
  ###opt= torch.optim.Adagrad(net.parameters())
  ###opt= torch.optim.RMSprop(net.parameters())
  #TEST: Learning rate scheduler:
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  #opt= torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.95, weight_decay=0.0)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  ##sch= torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
  #opt= torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=100, threshold=0.0001, threshold_mode='rel', verbose=True)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        #mode='min', factor=0.5, patience=100, cooldown=100,
        #threshold=-0.001, threshold_mode='rel',
        #verbose=True)
  opt= torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.95, weight_decay=0.005)
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  sch= ReduceLRAtCondition(opt,
         mode='gt', factor=0.5, patience=10, cooldown=200, threshold=0.001,
         verbose=True)
  #sch= torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0001, max_lr=0.1, step_size_up=100, mode='triangular2', verbose=True)
  #TEST: Stochastic Weight Averaging:
  #opt= torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.95, weight_decay=0.005)
  #swa_model= torch.optim.swa_utils.AveragedModel(net)
  #sch= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
  #swa_start= 160
  #swa_sch= torch.optim.swa_utils.SWALR(opt, swa_lr=0.05)

  loss_sub= torch.nn.MSELoss()
  loss= torch.nn.MSELoss()

  #opt= torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.95, weight_decay=0.0001)
  #loss= torch.nn.L1Loss()

  #opt= torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.95, weight_decay=0.005)
  #sch= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[125,150], gamma=0.5)
  #loss= torch.nn.HuberLoss(reduction='mean', delta=0.1*OUTFEAT_SCALE)

  #NOTE: Adjust the batch and epoch sizes.
  N_batch= 40
  N_epoch= 1000

  #Save the info into a file.
  now= time.localtime()
  time_stamp= '%04i.%02i.%02i-%02i.%02i.%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  info_filename= 'data_generated/log/{}_2-{}-{}-{}.info'.format(sqptn3,net.__class__.__name__,opt.__class__.__name__,time_stamp)
  with open(info_filename,'w') as fp:
    fp.write('net:\n'+str(net)+'\n\n')
    fp.write('opt:\n'+str(opt)+'\n\n')
    fp.write('sch:\n'+str(sch)+'\n'+str(sch.state_dict())+'\n\n')
    fp.write('loss:\n'+str(loss)+'\n'+str(loss.state_dict())+'\n\n')
    fp.write('loss_sub:\n'+str(loss_sub)+'\n'+str(loss_sub.state_dict())+'\n\n')
    fp.write('dataset_train: {} {}\n\n'.format(len(dataset_train),dataset_train.root))
    fp.write('dataset_test: {} {}\n\n'.format(len(dataset_test),dataset_test.root))
    fp.write('N_epoch,N_batch: {}, {}\n\n'.format(N_epoch,N_batch))
  print('saved info into:',info_filename)

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
  log_loss_sub_per_epoch= []
  log_loss_sub_test_per_epoch= []
  log_lr= []
  for i_epoch in range(N_epoch):
    log_loss_per_epoch.append(0.0)
    log_loss_sub_per_epoch.append(0.0)
    log_train_time.append(time.time())
    log_lr.append(opt.param_groups[0]['lr'])
    net.train()  # training mode; using dropout.
    for i_step, (batch_imgs1, batch_imgs2, batch_infeats, batch_outfeats) in enumerate(loader_train):
      b_imgs1= batch_imgs1
      b_imgs2= batch_imgs2
      b_infeats= batch_infeats
      b_outfeats= batch_outfeats
      b_imgs1,b_imgs2,b_infeats,b_outfeats= b_imgs1.to(device),b_imgs2.to(device),b_infeats.to(device),b_outfeats.to(device)

      opt.zero_grad()
      pred, pred_sub= net(b_imgs1,b_imgs2,b_infeats)
      #err_sub= loss_sub(pred_sub, b_infeats)  # must be (1. nn output, 2. target)
      err_sub= 0.1*loss_sub(pred_sub, b_infeats)  # must be (1. nn output, 2. target)
      err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)
      err_sub.backward(retain_graph=True)
      err.backward()
      opt.step()
      log_loss_per_epoch[-1]+= err.item()/len(loader_train)
      log_loss_sub_per_epoch[-1]+= err_sub.item()/len(loader_train)
      #print(i_epoch,i_step,err)
    log_train_time[-1]= time.time()-log_train_time[-1]

    #Test the network with the test data.
    log_loss_test_per_epoch.append(0.0)
    log_loss_sub_test_per_epoch.append(0.0)
    log_test_time.append(time.time())
    mse= 0.0  #MSE test.
    net.eval()  # evaluation mode; disabling dropout.
    with torch.no_grad():  # suppress calculating gradients.
      for i_step, (batch_imgs1, batch_imgs2, batch_infeats, batch_outfeats) in enumerate(loader_test):
        b_imgs1= batch_imgs1
        b_imgs2= batch_imgs2
        b_infeats= batch_infeats
        b_outfeats= batch_outfeats
        b_imgs1,b_imgs2,b_infeats,b_outfeats= b_imgs1.to(device),b_imgs2.to(device),b_infeats.to(device),b_outfeats.to(device)
        if swa_sch is None or i_epoch!=N_epoch-1:
          pred, pred_sub= net(b_imgs1,b_imgs2,b_infeats)
        else:
          pred, pred_sub= swa_model(b_imgs1,b_imgs2,b_infeats)
        err_sub= loss_sub(pred_sub, b_infeats)  # must be (1. nn output, 2. target)
        err= loss(pred, b_outfeats)  # must be (1. nn output, 2. target)
        log_loss_test_per_epoch[-1]+= err.item()/len(loader_test)
        log_loss_sub_test_per_epoch[-1]+= err_sub.item()/len(loader_test)
        mse+= torch.mean((pred-b_outfeats)**2).item()/len(loader_test)
        #print(i_epoch,i_step,err)
    log_test_time[-1]= time.time()-log_test_time[-1]
    if best_net_state is None or log_loss_test_per_epoch[-1]<best_net_loss:
      best_net_state= copy.deepcopy(net.state_dict())
      best_net_loss= log_loss_test_per_epoch[-1]
    if sch is not None:
      if swa_sch is None or i_epoch<=swa_start:
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
          #sch.step(log_loss_per_epoch[-1])
          sch.step(log_loss_test_per_epoch[-1])
          #sch.step(best_net_loss)
        elif isinstance(sch, ReduceLRAtCondition):
          N_maf= 20
          #schmetric= np.std(log_loss_test_per_epoch[-N_maf:])
          maf= [np.mean(log_loss_test_per_epoch[max(0,i+1-N_maf//2):i+1+N_maf//2]) for i in range(max(0,len(log_loss_test_per_epoch)-N_maf), len(log_loss_test_per_epoch))]
          schmetric= np.std((np.array(log_loss_test_per_epoch)[-len(maf):]-maf))
          sch.step(schmetric)
        else:
          sch.step()
      elif swa_sch is not None and i_epoch>swa_start:
        swa_model.update_parameters(net)
        swa_sch.step()
        if i_epoch==N_epoch-1:
          torch.optim.swa_utils.update_bn(loader_train, swa_model)
    print(i_epoch,log_loss_per_epoch[-1],log_loss_test_per_epoch[-1],mse,log_loss_sub_per_epoch[-1],log_loss_sub_test_per_epoch[-1])
  print('training time:',np.sum(log_train_time))
  print('testing time:',np.sum(log_test_time))
  print('best loss:',best_net_loss)

  #Recall the best net parameters:
  net.load_state_dict(best_net_state)

  #Save the model parameters into a file.
  #To load it: net.load_state_dict(torch.load(FILEPATH))
  torch.save(net.state_dict(), 'model_learned/cnn_{}_2-{}_{}.pt'.format(sqptn3,A_SIZE1,A_SIZE2))

  #Save the log into a file.
  log_filename= 'data_generated/log/{}_2-{}-{}-{}.log'.format(sqptn3,net.__class__.__name__,opt.__class__.__name__,time_stamp)
  with open(log_filename,'w') as fp:
    fp.write('#i_epoch train_time test_time loss_train loss_test\n')
    for i_epoch, (train_time, test_time, loss_train, loss_test, lr, loss_sub_train, loss_sub_test) in enumerate(zip(log_train_time, log_test_time, log_loss_per_epoch, log_loss_test_per_epoch, log_lr, log_loss_sub_per_epoch, log_loss_sub_test_per_epoch)):
      fp.write('{} {} {} {} {} {} {} {}\n'.format(i_epoch, train_time, test_time, loss_train, loss_test, lr, loss_sub_train, loss_sub_test))
  print('saved info into:',info_filename)
  print('saved log into:',log_filename)

  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  ax_lc.plot(range(len(log_loss_per_epoch)), log_loss_per_epoch, color='blue', label='loss_train')
  ax_lc.plot(range(len(log_loss_test_per_epoch)), log_loss_test_per_epoch, color='red', label='loss_test')
  ax_lc.plot(range(len(log_loss_sub_per_epoch)), log_loss_sub_per_epoch, color='green', label='loss_sub_train')
  ax_lc.plot(range(len(log_loss_sub_test_per_epoch)), log_loss_sub_test_per_epoch, color='purple', label='loss_sub_test')
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  fig1.tight_layout()

  net.eval()  # evaluation mode; disabling dropout.
  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_test)))
    img1,img2,in_feat,out_feat= dataset_test[i_data]
    pred,pred_sub= net(img1.view((1,)+img1.shape).to(device),img2.view((1,)+img2.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device))
    pred,pred_sub= pred.data.cpu().item()/OUTFEAT_SCALE, pred_sub.item()
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{}/in={:.3f}\nout={:.3f}\n/pred={:.3f}\n/pred_sub={:.3f}'.format(i_data,in_feat,out_feat,pred,pred_sub), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()

  fig3= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  for i in range(0,rows*cols):
    i_data= np.random.choice(range(len(dataset_train)))
    img1,img2,in_feat,out_feat= dataset_train[i_data]
    pred,pred_sub= net(img1.view((1,)+img1.shape).to(device),img2.view((1,)+img2.shape).to(device),in_feat.view((1,)+in_feat.shape).to(device))
    pred,pred_sub= pred.data.cpu().item()/OUTFEAT_SCALE, pred_sub.item()
    img1= ((img1+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img2= ((img2+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    img= torch.cat((img1,img2), axis=2)
    in_feat= in_feat.item()
    out_feat= out_feat.item()/OUTFEAT_SCALE
    ax= fig3.add_subplot(rows, cols, i+1)
    ax.set_title('train#{}/in={:.3f}\nout={:.3f}\n/pred={:.3f}\n/pred_sub={:.3f}'.format(i_data,in_feat,out_feat,pred,pred_sub), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig3.tight_layout()

  plt.show()
  #'''
