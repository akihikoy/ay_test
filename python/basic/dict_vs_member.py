#!/usr/bin/python3
import time

class TTest:
  def __init__(self):
    self.A= 1.0
    self.D= {'A':1.0}

if __name__=='__main__':
  test= TTest()
  d= {'A':1.0}

  dt= {'member':0.0, 'mem_dict':0.0, 'dict':0.0}
  N1= 100
  N2= 100000
  for i in range(N1):
    t0= time.time()
    for j in range(N2):
      test.A= 1.0001*test.A + 1.0
    t1= time.time()
    dt['member']+= t1-t0

    t0= time.time()
    for j in range(N2):
      test.D['A']= 1.0001*test.D['A'] + 1.0
    t1= time.time()
    dt['mem_dict']+= t1-t0

    t0= time.time()
    for j in range(N2):
      d['A']= 1.0001*d['A'] + 1.0
    t1= time.time()
    dt['dict']+= t1-t0

  print(dt)

