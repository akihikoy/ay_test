#!/usr/bin/python
from numpy import *
from math import *

def Rot(p,theta):
  c= cos(theta)
  s= sin(theta)
  p2= array([p[0]*c-p[1]*s, p[0]*s+p[1]*c])
  return p2

def DrawLineRecursively(x1,x2,p_list,n,N):
  if n==N:
    print ' '.join(map(str,x1))
    print ' '.join(map(str,x2))
  else:
    p0= p_list[0]
    d1= p_list[-1]-p_list[0]
    d2= x2-x1
    f= linalg.norm(d2)/linalg.norm(d1)
    d= x1-p0
    theta= acos(dot(d1,d2)/(linalg.norm(d1)*linalg.norm(d2)))
    if cross(d1,d2)<0: theta=-theta
    pa= p_list[0]
    for pb in p_list[1:]:
      pa2= f*Rot(pa-p0,theta)+d
      pb2= f*Rot(pb-p0,theta)+d
      DrawLineRecursively(pa2,pb2,p_list,n+1,N)
      pa= pb

#p_list= array([[0,0],[1,0],[1.5,sqrt(3)/2.0],[2,0],[3,0]])
p_list= array([[0,0],[1,0],[1,1],[2,1],[2,0],[3,0]])
#p_list= array([[0,0],[1,0],[1,1],[2,1],[2,-1],[3,-1],[3,0],[4,0]])
#p_list= array([[0,0],[1,0],[1,2],[2,2],[2,0],[3,0]])
#p_list= array([[0,0],[1,0],[1.5,2],[2,0],[3,0]])

#DrawLineRecursively(array([0,0]),array([1,0]),p_list,0,5)

NN=8
for i in range(0,NN):
  p1= Rot(array([0,1]),2.0*pi/float(NN)*i)
  p2= Rot(array([0,1]),2.0*pi/float(NN)*(i+1))
  DrawLineRecursively(p1,p2,p_list,0,4)
