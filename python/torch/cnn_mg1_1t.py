#!/usr/bin/python3
#\file    cnn_mg1_1t.py
#\brief   Test the learned model with cnn_mg1_1.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.27, 2021
from cnn_mg1_1 import *

if __name__=='__main__':
  import sys
  model_file= sys.argv[1]

  #dataset_test= MG1Dataset(transform=GetDataTransforms('train'), train=False)
  dataset_test= MG1Dataset(transform=GetDataTransforms('eval'), train=False)

  #NOTE: Switch the NN definition.
  #Setup a neural network.
  net= TAlexNet(img_shape=dataset_test[0][0].shape)

  #NOTE: Switch the device.
  #device= 'cpu'
  device= 'cuda'  # recommended to check by torch.cuda.is_available()
  net= net.to(device)

  if model_file is not None:
    net.load_state_dict(torch.load(model_file))

  print(net)

  net.eval()  # evaluation mode; disabling dropout.
  fig2= plt.figure(figsize=(8,8))
  rows,cols= 5,4
  log_times= []
  for i in range(rows*cols):
    t0= time.time()
    i_data= np.random.choice(range(len(dataset_test)))
    img,label= dataset_test[i_data]
    label= label.item()/LABEL_SCALE
    pred= net(img.view((1,)+img.shape).to(device)).data.cpu().item()/LABEL_SCALE
    log_times.append(time.time()-t0)
    img= ((img+1.0)*(255.0/2.0)).type(torch.uint8)  #Convert image for imshow
    ax= fig2.add_subplot(rows, cols, i+1)
    ax.set_title('test#{0}\n/l={1:.4f}\n/pred={2:.4f}'.format(i_data,label,pred), fontsize=8)
    ax.imshow(img.permute(1,2,0))
  fig2.tight_layout()
  #print(log_times)
  print('Average computation time per image: {:.5f}s'.format(np.mean(log_times)))

  plt.show()
  #'''
