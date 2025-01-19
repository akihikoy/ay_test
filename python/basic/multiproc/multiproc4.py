#!/usr/bin/python3
#\file    multiproc4.py
#\brief   multiprocessing test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.06, 2018
import multiprocessing as mp
import time

def Func(sleep):
  t0= time.time()
  print('time',t0)
  time.sleep(sleep)
  print('slept',sleep,time.time()-t0)

if __name__=='__main__':
  ps= [mp.Process(target=Func, args=(10.0*((i+1)*0.02),)) for i in range(5)]
  for p in ps:  p.start()
  for p in ps:  p.join()
