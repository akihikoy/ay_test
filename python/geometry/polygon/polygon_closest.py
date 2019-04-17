#!/usr/bin/python
import numpy as np
import numpy.linalg as la

def Vec(a):
  return np.array(a)

#Closest point on a line (p1,p2) from a reference point
def LineClosestPoint(p1, p2, point_ref):
  a= Vec(p2)-Vec(p1)
  t_max= la.norm(a)
  a= a/t_max
  t= np.dot(Vec(point_ref)-p1, a)
  if t>=t_max:  return p2
  if t<=0:      return p1
  return Vec(p1) + t*a

#Closest point on a polygon from a reference point
def PolygonClosestPoint(points, point_ref):
  if len(points)==0:  return None
  if len(points)==1:  return points[0]
  p_closest= None
  d_closest= 1.0e20
  N= len(points)
  for i in range(N):
    p= LineClosestPoint(points[i], points[i+1 if i+1!=N else 0], point_ref)
    d= la.norm(Vec(point_ref)-Vec(p))
    if p_closest==None or d<d_closest:
      p_closest= p
      d_closest= d
  return p_closest, d_closest


import random
from gen_data import *
points= Gen3d_01()
#points= Gen3d_02()
#points= Gen3d_11()
#points= Gen3d_12()
#points= Gen3d_13()

fp= file('/tmp/orig.dat','w')
for p in points:
  fp.write(' '.join(map(str,p))+'\n')
fp.close()

p_ref= [0.4*(random.random()-0.5),0.4*(random.random()-0.5),0.2*(random.random())]
p,d= PolygonClosestPoint(points, p_ref)

fp= file('/tmp/res.dat','w')
fp.write(' '.join(map(str,p))+'\n')
fp.close()

fp= file('/tmp/res2.dat','w')
fp.write(' '.join(map(str,p_ref))+'\n')
fp.close()

#fp1= file('/tmp/res.dat','w')
#fp2= file('/tmp/res2.dat','w')
#for i in range(20):
  #p_ref= [0.4*(random.random()-0.5),0.4*(random.random()-0.5),0.2*(random.random())]
  #p,d= PolygonClosestPoint(points, p_ref)
  #fp1.write(' '.join(map(str,p))+'\n')
  #fp2.write(' '.join(map(str,p_ref))+'\n')
#fp1.close()
#fp2.close()
