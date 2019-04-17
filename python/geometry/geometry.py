#!/usr/bin/python
#\file    geometry.py
#\brief   Geometry library
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.07, 2015
import math
import numpy as np
import numpy.linalg as la
from quaternion import QToRot, RotToQ

#Convert a pose, x,y,z,quaternion(qx,qy,qz,qw) to pos (x,y,z) and 3x3 rotation matrix
def XToPosRot(x):
  p = np.array(x[0:3])
  R = tf.transformations.quaternion_matrix(x[3:7])[:3,:3]
  return p, R

#Convert pos p=(x,y,z) and 3x3 rotation matrix R to a pose, x,y,z,quaternion(qx,qy,qz,qw)
def PosRotToX(p,R):
  M = tf.transformations.identity_matrix()
  M[:3,:3] = R
  x = list(p)+[0.0]*4
  x[3:7] = tf.transformations.quaternion_from_matrix(M)
  return x


if __name__=='__main__':
  pass
