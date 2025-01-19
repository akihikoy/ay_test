#!/usr/bin/python3
#\file    mp_with_lock.py
#\brief   multiprocessing with a locker.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.02, 2020
'''
Summary.
Compared:
  1. threading with threading.RLock
  2. multiprocessing with mp.RLock
  3. multiprocessing with threading.RLock
Result:
  1. The locker object locked the block.
  2. The locker object locked the block.
  3. The locker object did not lock the block.
See the codes with #NOTE.
'''

import multiprocessing as mp
import threading
import queue
import random,sys,time

def SubFunc(pid, queue_cmd, queue_out, print_locker, parameter):
  print('Started:',pid,parameter)
  y= 0
  for x in range(int(parameter*1e4)):
    with print_locker:
      y+= x
      time.sleep(0.001)
      if x%100==0:
        print(pid, y)
        sys.stdout.flush()
    try:
      cmd= queue_cmd.get(block=False)
      if cmd=='stop':  break
    except queue.Empty:
      pass
  queue_out.put((pid, parameter, x, y))

#Multiprocessing version.
def MainMP():
  queue_cmd= mp.Queue()
  queue_out= mp.Queue()
  #print_locker= threading.RLock()  #NOTE
  print_locker= mp.RLock()  #NOTE
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in range(8):
    p= random.random()
    new_proc= mp.Process(target=SubFunc, args=(pid,queue_cmd,queue_out,print_locker,p))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  pid_out,parameter,x,y= queue_out.get()
  processes[pid_out].join()
  del processes[pid_out]
  results.append((pid_out,parameter,x,y))
  print('Finished:',results[-1])

  for _ in processes:  queue_cmd.put('stop')
  for proc in list(processes.values()):  proc.join()

  print('Results:',results)

#Thread version.
def MainTh():
  queue_cmd= queue.Queue()
  queue_out= queue.Queue()
  print_locker= threading.RLock()
  pid= 0  #thread ID
  threads= {}  #pid:thread
  for _ in range(8):
    p= random.random()
    new_thread= threading.Thread(target=SubFunc, args=(pid,queue_cmd,queue_out,print_locker,p))
    threads[pid]= new_thread
    threads[pid].start()
    pid+= 1

  results= []
  pid_out,parameter,x,y= queue_out.get()
  threads[pid_out].join()
  del threads[pid_out]
  results.append((pid_out,parameter,x,y))
  print('Finished:',results[-1])

  for _ in threads:  queue_cmd.put('stop')
  for thread in list(threads.values()):  thread.join()

  print('Results:',results)


if __name__=='__main__':
  #MainTh()  #NOTE
  MainMP()  #NOTE


