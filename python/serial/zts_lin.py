#!/usr/bin/python
#\file    zts_lin.py
#\brief   Reading values simultaneously from an IMADA ZTS force sensor and a linear encoder on Arduino.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.19, 2021
from IMADA_ZTS2 import TZTS
from encoder1 import TLinearEncoder
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
#import threading
#import copy
import multiprocessing as mp
import Queue

def PlotLoop(queue_data):
  data= []
  N1,N2= 1000,1000
  plt.rcParams['keymap.quit'].append('q')
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
  ax1b= ax1.twinx()
  t0= time.time()
  while True:
    try:
      for i in range(100):
        value= queue_data.get(block=False)
        data.append(value)
    except Queue.Empty:
      pass
    if len(data)==0:
      time.sleep(0.02)
      continue
    d= np.array(data)
    ax1.cla()
    ax1b.cla()
    ax1.plot(d[-N1:,0]-t0, d[-N1:,1], color='blue', linewidth=1, label='Force')
    ax1b.plot(d[-N1:,0]-t0, d[-N1:,2], color='red', linewidth=1, label='Position')

    ax2.cla()
    ax2.plot(d[-N2:,2], d[-N2:,1], color='green', linewidth=1)

    ax1.set_title('Time-Force,Position')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')
    #ax1.set_ylim(bottom=-1.2,top=1.2)
    ax1.legend()
    ax1b.set_ylabel('Position [mm]')
    ax1b.legend(loc='upper right', bbox_to_anchor=(1.0,0.8))
    ax2.set_title('Position-Force')
    ax2.set_xlabel('Position [mm]')
    ax2.set_ylabel('Force [N]')
    plt.pause(0.02)


if __name__=='__main__':
  log_file_name= sys.argv[1] if len(sys.argv)>1 else None
  dev_zts= sys.argv[2] if len(sys.argv)>2 else '/dev/ttyS6'  #COM6(Win); /dev/ttyACM0
  dev_lin= sys.argv[3] if len(sys.argv)>3 else '/dev/ttyS5'  #COM5(Win); /dev/ttyACM0
  with_plot= bool(sys.argv[4]) if len(sys.argv)>4 else True

  if with_plot:
    queue_plot= mp.Queue()
    proc_plot= mp.Process(target=PlotLoop, args=(queue_plot,))
    proc_plot.start()

  dt_sleep= 0.01
  zts= TZTS(dev_zts, cnt=True)
  lin= TLinearEncoder(dev_lin)
  try:
    log_fp= open(log_file_name, 'w') if log_file_name else None
    while True:
      t= time.time()
      n_zts, n_lin, v_zts, u_zts, v_lin= zts.N, lin.N, zts.Value, zts.Unit, lin.Value
      if not isinstance(v_zts,float):  print 'ZTS invalid value:',zts.Raw; time.sleep(dt_sleep); continue
      if not isinstance(v_lin,float):  print 'LIN invalid value:',lin.Raw; time.sleep(dt_sleep); continue
      print '{n_zts:06d} {n_lin:06d} Latest: {v_zts:9.04f}{u_zts} {v_lin:9.04f}'.format(n_zts=n_zts, n_lin=n_lin, v_zts=v_zts, u_zts=u_zts, v_lin=v_lin)

      if log_fp:
        log_fp.write('{} {} {} {}\n'.format(t, v_zts, u_zts, v_lin))
      if with_plot:
        queue_plot.put([t, v_zts, v_lin])
      time.sleep(dt_sleep)

  except KeyboardInterrupt:
    pass

  finally:
    zts.Stop()
    lin.Stop()
    if log_fp:
      log_fp.close()
      print 'Logged into:',log_file_name
    if with_plot:
      proc_plot.terminate()
