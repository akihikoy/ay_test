#!/usr/bin/python
import yaml
import math,random
import numpy as np
import numpy.linalg as la
from pca2 import TPCA

def Gen3d_01():
  points= yaml.load(file('../data/polygon.yaml').read())['polygon']
  return points

def Gen3d_02():
  points= yaml.load(file('../data/polygon2.yaml').read())['polygon']
  return points

def Gen3d_11():
  theta= 0.0
  points= []
  while theta<2.0*math.pi:
    p= [0]*3
    p[0]= 1.0+(0.5+0.2*random.random())*math.cos(theta)
    p[1]= 1.0+(0.7+0.3*random.random())*math.sin(theta)
    p[2]= 1.0+0.2*p[1]+0.2*random.random()
    points.append(p)
    if theta<0.25*math.pi:
      theta+= 0.005
    else:
      theta+= 0.1
  return points

def Gen3d_12():
  theta= 1.5*math.pi
  points= []
  while theta<3.0*math.pi:
    p= [0]*3
    p[0]= 1.0+(0.5+0.2*random.random())*math.cos(theta)
    p[1]= 1.0+(0.7+0.3*random.random())*math.sin(theta)
    p[2]= 1.0+0.2*p[1]+0.2*random.random()
    points.append(p)
    if theta<0.25*math.pi:
      theta+= 0.005
    else:
      theta+= 0.1
  return points

def Gen3d_13():
  theta= 0.0
  points= []
  while theta<2.0*math.pi:
    p= [0]*3
    p[0]= (0.5+0.2*random.random())*math.cos(theta)
    p[1]= (0.7+0.3*random.random())*math.sin(theta)
    p[2]= 1.0+0.2*random.random()
    points.append(p)
    if theta<0.25*math.pi:
      theta+= 0.005
    else:
      theta+= 0.1
  return points


def To2d(points):
  pca= TPCA(points,calc_projected=True)
  return pca.Projected[:,[0,1]]

def To2d2(points):
  return np.array([[p[0],p[1]] for p in points])


if __name__=='__main__':
  #points= Gen3d_01()
  #points= Gen3d_02()
  points= Gen3d_11()
  #points= Gen3d_12()
  #points= Gen3d_13()

  #points= To2d(Gen3d_01())
  #points= To2d(Gen3d_02())
  #points= To2d(Gen3d_11())
  #points= To2d(Gen3d_12())
  #points= To2d(Gen3d_13())

  fp= file('/tmp/orig.dat','w')
  for p in points:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  print 'Plot by'
  if len(points[0])==2:
    print "qplot -x /tmp/orig.dat w l"
  elif len(points[0])==3:
    print "qplot -x -3d -s 'set ticslevel 0' /tmp/orig.dat w l"

