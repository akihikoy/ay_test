#!/usr/bin/python3
#\file    multiproc3.py
#\brief   multiprocessing test 3
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.02, 2016
import multiprocessing as mp
import queue
import time

class TTest:
  def __init__(self,q,n):
    self.q= q
    self.n= n

  def Func1(self,queue_cmd,queue_out,p):
    assert(p>0)
    s= 0
    for x in range(0,self.q,p):
      s+= x
      if x%1000000==0:  #This check needs a bigger computation time
        try:
          cmd= queue_cmd.get(block=False)
          if cmd=='stop':
            p= -p
            break
        except queue.Empty:
          pass
    queue_out.put((p,s))

  def Func1M(self):
    queue_cmd= mp.Queue()
    queue_out= mp.Queue()
    ps= [mp.Process(target=self.Func1, args=(queue_cmd,queue_out,pp+1)) for pp in range(self.n)]
    for p in ps:  p.start()
    for i in range(4):
      p,s= queue_out.get()
      print('{p}: {s}'.format(p=p,s=s))
    time.sleep(1.0)
    for i in range(self.n-4):
      queue_cmd.put('stop')
    for i in range(self.n-4):
      p,s= queue_out.get()
      print('Stopped: {p}: {s}'.format(p=p,s=s))
    for p in ps:  p.join()

if __name__=='__main__':
  t= TTest(q=10**8,n=8)
  t.Func1M()

