#!/usr/bin/python3
import numpy as np
import numpy.linalg as la
import math
from rodrigues import *

def RFromAxisAngle(axis,angle):
  w= angle * np.array(axis) / la.norm(axis)
  return Rodrigues(w)

#InvRodrigues= InvRodrigues0

if __name__=='__main__':
  import random,time
  rand1= lambda: 2.0*(random.random()-0.5)

  t_start= time.time()
  max_error= 0.0
  for i in range(2000):
    axis= [rand1() for d in range(3)]
    angle= math.pi*rand1()
    R= RFromAxisAngle(axis, angle)
    w= InvRodrigues(R)
    theta= la.norm(w)
    error= min(theta-angle,theta+angle)
    max_error= max(max_error,error)
    #print 'theta=',theta/math.pi*180.0,'angle=',angle/math.pi*180.0,'error=',error

  print('max_error=',max_error)

  R= np.array([[-0.9999296977757071, 4.89652762780679e-12, -0.011857466263212654], [-4.896194545289173e-12, -1.0, -5.795627477184499e-14], [-0.011857466263212681, 5.5511151231257815e-17, 0.9999296977757071]])
  w= InvRodrigues(R)
  print('R=',R)
  print('theta=',la.norm(w)/math.pi*180.0)

  print('comptation time:',time.time()-t_start)
