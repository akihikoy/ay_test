#!/usr/bin/python
import random

#Search x such that fobj(x) is not None and x is in bound
def InitialGuess(bound, fobj, max_count=None):
  assert(len(bound)==2)
  assert(len(bound[0])==len(bound[1]))
  xmin= bound[0]
  xmax= bound[1]
  dim= len(xmin)
  dx= [xmax[d]-xmin[d] for d in range(dim)]
  while max_count==None or max_count>0:
    x= [xmin[d]+dx[d]*random.random() for d in range(dim)]
    if fobj(x) is not None:
      #print max_count
      return x
    if max_count is not None:  max_count-= 1
  return None

bound= [[-1.0,-0.5],[1.0,0.5]]
def fobj1(x):
  if (x[0]+0.5)**2+(x[1]+0.5)**2<0.5**2:  return None
  return x[0]+x[1]

for i in range(10000):
  x= InitialGuess(bound, fobj1, max_count=1000000)
  f= fobj1(x)
  print x[0],x[1],f
