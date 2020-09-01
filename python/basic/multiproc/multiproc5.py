#!/usr/bin/python
#\file    multiproc5.py
#\brief   multiprocessing test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.01, 2020
import multiprocessing as mp
import Queue
import random

def SubFunc(pid, queue_cmd, queue_out, parameter):
  print 'Started:',pid,parameter
  y= 0
  for x in xrange(int(parameter*1e7)):
    y+= x
    try:
      cmd= queue_cmd.get(block=False)
      if cmd=='stop':  break
    except Queue.Empty:
      pass
  queue_out.put((pid, parameter, x, y))

def Main1():
  queue_cmd= mp.Queue()
  queue_out= mp.Queue()
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in xrange(8):
    p= random.random()
    new_proc= mp.Process(target=SubFunc, args=(pid,queue_cmd,queue_out,p))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  while len(processes)>0:
    pid_out,parameter,x,y= queue_out.get()
    processes[pid_out].join()
    del processes[pid_out]
    results.append((pid_out,parameter,x,y))
    print 'Finished:',results[-1]

def Main2():
  queue_cmd= mp.Queue()
  queue_out= mp.Queue()
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in xrange(8):
    p= random.random()
    new_proc= mp.Process(target=SubFunc, args=(pid,queue_cmd,queue_out,p))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  pid_out,parameter,x,y= queue_out.get()
  processes[pid_out].join()
  del processes[pid_out]
  results.append((pid_out,parameter,x,y))
  print 'Finished:',results[-1]

  for _ in processes:  queue_cmd.put('stop')
  while len(processes)>0:
    pid_out,parameter,x,y= queue_out.get()
    processes[pid_out].join()
    del processes[pid_out]
    results.append((pid_out,parameter,x,y))
    print 'Finished:',results[-1]

  #for _ in processes:  queue_cmd.put('stop')
  #for _ in processes:  queue_out.get()
  #for proc in processes.values():  proc.join()

def Main3():
  queue_cmd= mp.Queue()
  queue_out= mp.Queue()
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in xrange(8):
    p= random.random()
    new_proc= mp.Process(target=SubFunc, args=(pid,queue_cmd,queue_out,p))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  pid_out,parameter,x,y= queue_out.get()
  processes[pid_out].join()
  del processes[pid_out]
  results.append((pid_out,parameter,x,y))
  print 'Finished:',results[-1]

  #NOTE: When we use terminate, the queues may become corrupted.
  for proc in processes.values():  proc.terminate()
  for proc in processes.values():  proc.join()


if __name__=='__main__':
  #Main1()
  #Main2()
  Main3()
