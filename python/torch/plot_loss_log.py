#!/usr/bin/python3
#\file    plot_loss_log.py
#\brief   Plot loss curves in log file.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.02, 2021
#!/usr/bin/python3
#\file    lr_sch_1.py
#\brief   Test learning schedulers of PyTorch.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.30, 2021
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cols
import time
import sys

if __name__=='__main__':
  log_file_names= sys.argv[1:]

  log_train_time= []
  log_test_time= []
  log_loss_per_epoch= []
  log_loss_test_per_epoch= []
  for log_file_name in log_file_names:
    with open(log_file_name,'r') as fp:
      log_train_time.append([])
      log_test_time.append([])
      log_loss_per_epoch.append([])
      log_loss_test_per_epoch.append([])
      while True:
        line= fp.readline()
        if not line:  break
        if line[0]=='#':  continue
        i_epoch,train_time,test_time,loss_train,loss_test= map(float,line.split()[:5])
        log_train_time[-1].append(train_time)
        log_test_time[-1].append(test_time)
        log_loss_per_epoch[-1].append(loss_train)
        log_loss_test_per_epoch[-1].append(loss_test)

  grad= {'blue':[plt_cols.hsv_to_rgb((0.56823266, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'red':[plt_cols.hsv_to_rgb((0.99904762, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'green':[plt_cols.hsv_to_rgb((0.33333333, s, 0.9)) for s in np.linspace(1.0,0.3,5)],
         'orange':[plt_cols.hsv_to_rgb((0.07814661, s, 0.9)) for s in np.linspace(1.0,0.3,5)]}
  fig1= plt.figure()
  ax_lc= fig1.add_subplot(1,1,1)
  for i,(loss_per_epoch,loss_test_per_epoch) in enumerate(zip(log_loss_per_epoch,log_loss_test_per_epoch)):
    ax_lc.plot(range(len(loss_per_epoch)), loss_per_epoch, color=grad['blue'][(i)%len(grad['blue'])], label='[{}]loss_train'.format(i))
    ax_lc.plot(range(len(loss_test_per_epoch )), loss_test_per_epoch , color=grad['red'][(i)%len(grad['red'])], label='[{}]loss_test'.format(i))
  ax_lc.set_title('Loss curve')
  ax_lc.set_xlabel('epoch')
  ax_lc.set_ylabel('loss')
  ax_lc.set_yscale('log')
  ax_lc.legend()
  fig1.tight_layout()

  plt.show()
