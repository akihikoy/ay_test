#!/usr/bin/python3
import math
import copy
import numpy as np

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

#Convert radian to [-pi,pi)
def AngleMod1(q):
  return Mod(q+math.pi,math.pi*2.0)-math.pi

#Convert radian to [0,2*pi)
def AngleMod2(q):
  return Mod(q,math.pi*2.0)

#Remove radian jumping in traj
def AngleTrajSmoother(traj):
  q_prev= np.array(traj[0])
  q_offset= np.array([0]*len(q_prev))
  for q in traj:
    q_diff= np.array(q) - q_prev
    for d in range(len(q_prev)):
      if q_diff[d]<-math.pi:  q_offset[d]+=1
      elif q_diff[d]>math.pi:  q_offset[d]-=1
    q_prev= copy.deepcopy(q)
    q[:]= q+q_offset*2.0*math.pi

traj=[]
t= -10
while t<10:
  q1= AngleMod1(t)
  q2= AngleMod2(t*t)
  q3= AngleMod1(5.0*math.sin(t))
  traj.append([t,q1,q2,q3])
  t+=0.01

traj_old= copy.deepcopy(traj)
AngleTrajSmoother(traj)
for a,b in zip(traj_old,traj):
  print(' '.join(map(str,a)),' '.join(map(str,b)))

