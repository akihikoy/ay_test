#!/usr/bin/python3
#\file    multiproc1.py
#\brief   multiprocessing test
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.02, 2016
import multiprocessing as mp

def Func1(p):
  assert(p>0)
  s= 0
  for x in range(0,int(1e8),p):
    s+= x
  return p,s

if __name__=='__main__':
  n= 8

  #SINGLE PROCESS
  #for pp in xrange(n):
    #p,s= Func1(pp+1)
    #print '{p}: {s}'.format(p=p,s=s)

  #n-PROCESSES
  pool= mp.Pool(n)
  callback= pool.map(Func1, (pp+1 for pp in range(n)))
  for p,s in callback:
    print('{p}: {s}'.format(p=p,s=s))
