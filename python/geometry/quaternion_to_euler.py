#!/usr/bin/python
#import tf
import rostf
import math
import numpy as np
import numpy.linalg as la

def YZXEulerFromQuaternion(q):
  #return rostf.euler_from_quaternion(q, axes='syzx')
  return rostf.euler_from_quaternion(q, axes='ryzx')


def QFromAxisAngle(axis,angle):
  axis= axis / la.norm(axis)
  return rostf.quaternion_about_axis(angle,axis)

#Quaternion to 3x3 rotation matrix
def QToRot(q):
  return rostf.quaternion_matrix(q)[:3,:3]

#3x3 rotation matrix to quaternion
def RotToQ(R):
  M = rostf.identity_matrix()
  M[:3,:3] = R
  return rostf.quaternion_from_matrix(M)

def RFromAxisAngle(axis,angle):
  return QToRot(QFromAxisAngle(axis,angle))
  return rotation_matrix(angle,axis)

if __name__=='__main__':
  import random
  rand1= lambda: 2.0*(random.random()-0.5)
  q= QFromAxisAngle([rand1() for d in range(3)], math.pi*rand1())
  #q= [0.05240207810290061, -0.620853105352924, -0.6535491923757895, 0.4297311914779772]
  q= np.array(q)
  e= YZXEulerFromQuaternion(q)
  print 'q=',q
  print 'YZX-Euler=',e
  rotY= RFromAxisAngle([0.,1.,0.],e[0])
  rotZ= RFromAxisAngle([0.,0.,1.],e[1])
  rotX= RFromAxisAngle([1.,0.,0.],e[2])
  #print 'rotX',rotX
  #print 'rotY',rotY
  #print 'rotZ',rotZ
  #E= RotToQ( np.dot( rotX, np.dot(rotZ, rotY ) ) )
  E= RotToQ( np.dot( rotY, np.dot(rotZ, rotX ) ) )
  print 'Q from YZX euler=', E
  print 'q==E?', np.allclose(q,E)
  print 'QToRot(q)==QToRot(E)?', np.allclose(QToRot(q),QToRot(E))

