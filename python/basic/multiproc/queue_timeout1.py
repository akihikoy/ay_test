#!/usr/bin/python3
#\file    queue_timeout1.py
#\brief   Test timeout of queue.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.20, 2021
import multiprocessing as mp
import queue
import random, copy
import os,signal

def SubFunc(pid, queue_cmd, queue_out, parameter):
  print('Started:',pid,parameter,id(parameter))
  Y= []
  for x in range(int(parameter[0]*1e6)):
    if x%100==0:  Y.append(x)
    else:         Y[-1]+= x
    try:
      cmd= queue_cmd.get(block=False)
      if cmd=='stop':  break
    except queue.Empty:
      pass
  queue_out.put((pid, parameter, x, Y[-1]))
  return pid, parameter, x, Y[-1]

def Main1(param_list):
  param_list= copy.deepcopy(param_list)
  max_proc= 10

  queue_cmd= mp.Queue()
  queue_out= mp.Queue()
  pid= 0  #process ID
  results= []
  processes= {}  #pid:process
  try:
    while len(param_list)>0 or len(processes)>0:
      while len(param_list)>0 and len(processes)<max_proc:
        p= param_list.pop(0)
        print('SubFunc request with id(p)',id(p))
        new_proc= mp.Process(target=SubFunc, args=(pid,queue_cmd,queue_out,p))
        processes[pid]= new_proc
        processes[pid].start()
        pid+= 1
      pid_out,parameter,x,y= queue_out.get(timeout=0.15)
      processes[pid_out].join(timeout=0.15)
      exitcode= processes[pid_out].exitcode
      del processes[pid_out]
      results.append((pid_out,parameter,x,y))
      print('Finished:',results[-1],exitcode)
  except queue.Empty:
    print('Timeout.')
    for proc in processes.values():
      print('  Terminating',proc.pid,proc.exitcode)
      #proc.terminate()
      os.kill(proc.pid,signal.SIGKILL)
    time.sleep(0.002)
    print('Termination double check.')
    for proc in processes.values():
      print('  Proc',proc.pid,proc.is_alive())

  print('Results:')
  for res in results:
    print(res, id(res[1]))


if __name__=='__main__':
  param_list= [[random.random(),0] for _ in range(10)]
  import time
  t_start= time.time()
  Main1(param_list)
  print('Main1 time:',time.time()-t_start)
