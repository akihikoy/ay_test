#!/usr/bin/python
import numpy as np
import numpy.linalg as la
import math
from rodrigues import *

def RFromAxisAngle(axis,angle):
  w= angle * np.array(axis) / la.norm(axis)
  return Rodrigues(w)


if __name__=='__main__':
  import random
  rand1= lambda: 2.0*(random.random()-0.5)

  for i in range(20):
    R= RFromAxisAngle([rand1() for d in range(3)], math.pi*rand1())
    w= InvRodrigues(R)
    theta= la.norm(w)
    print 'theta=',theta/math.pi*180.0


