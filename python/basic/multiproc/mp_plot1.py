#!/usr/bin/python
#\file    mp_plot1.py
#\brief   Plot with multiprocessing;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.28, 2021
import multiprocessing as mp
import Queue
import numpy as np
import matplotlib.pyplot as plt
import time

def PlotLoop(queue_data):
  plt.rcParams['keymap.quit'].append('q')
  t= 0.0
  data= []
  while True:
    #try:
      #cmd= queue_cmd.get(block=False)
      #if cmd=='stop':  break
    #except Queue.Empty:
      #pass
    try:
      for i in range(100):
        value= queue_data.get(block=False)
        data.append(value)
      #print 'RCV',value
    except Queue.Empty:
      pass
    if len(data)==0:  continue
    d= np.array(data)
    plt.cla()
    plt.scatter(d[:,0], d[:,1], color='blue', label='random')
    plt.title('Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.pause(0.05)

if __name__=='__main__':
  #queue_cmd= mp.Queue()
  queue_data= mp.Queue()
  proc= mp.Process(target=PlotLoop, args=(queue_data,))
  proc.start()
  t= 0.0
  try:
    while True:
      queue_data.put([t, np.sin(t)])
      print [t, np.sin(t)]
      t+= 0.1
      time.sleep(0.01)

  except KeyboardInterrupt:
    pass

  finally:
    #queue_cmd.put('stop')
    #proc.join(timeout=5)
    proc.terminate()
