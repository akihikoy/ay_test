#!/usr/bin/python3
#\file    mp_opt.py
#\brief   Optimization with multiprocess.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.02, 2020
import multiprocessing as mp
import queue
import random

from funcs import Func,Plot
import scipy.optimize
import numpy as np
import time

#Single-process optimizer.
def SubOpt(pid, queue_out, fkind):
  print('Started:',pid,fkind)
  xmin= [-10.,-10.]
  xmax= [10.,10.]
  f_slow= lambda x:[Func(x,fkind),time.sleep(0.005)][0]  #Simulate a complicated function.

  # Minimize the function f
  tol= 1.0e-5
  res= scipy.optimize.differential_evolution(f_slow, np.array([xmin,xmax]).T, strategy='best1bin', maxiter=30, popsize=10, tol=tol, mutation=(0.5, 1), recombination=0.7, seed=int(random.random()*1000000))
  queue_out.put((pid, res.x, f_slow(res.x)))

'''
Run 20 optimizers in parallel,
and wait for the all processes to finish.
'''
def Main1(fkind):
  queue_out= mp.Queue()
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in range(20):
    p= random.random()
    new_proc= mp.Process(target=SubOpt, args=(pid,queue_out,fkind))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  while len(processes)>0:
    pid_out,x,score= queue_out.get()
    processes[pid_out].join()
    del processes[pid_out]
    results.append((pid_out,x,score))
    print('Finished:',results[-1])

  results.sort(key=lambda r:r[-1])
  print('Best=',results[0])
  return results

'''
Run 20 optimizers in parallel,
and wait only for num_sol processes to finish,
and then terminate the running processes.
'''
def Main2(fkind, num_sol=3):
  queue_out= mp.Queue()
  pid= 0  #process ID
  processes= {}  #pid:process
  for _ in range(20):
    p= random.random()
    new_proc= mp.Process(target=SubOpt, args=(pid,queue_out,fkind))
    processes[pid]= new_proc
    processes[pid].start()
    pid+= 1

  results= []
  while len(processes)>0:
    pid_out,x,score= queue_out.get()
    processes[pid_out].join()
    del processes[pid_out]
    results.append((pid_out,x,score))
    print('Finished:',results[-1])
    if len(results)>=num_sol:  break

  for proc in list(processes.values()):  proc.terminate()
  for proc in list(processes.values()):  proc.join()

  results.sort(key=lambda r:r[-1])
  print('Best=',results[0])
  return results

if __name__=='__main__':
  import sys
  fkind= int(sys.argv[1]) if len(sys.argv)>1 else 5

  #results= Main1(fkind)
  results= Main2(fkind, num_sol=5)
  xmin= [-1.,-1.]
  xmax= [2.,3.]
  Plot(xmin,xmax,lambda x:Func(x,fkind),x_points=[x for _,x,_ in results])

