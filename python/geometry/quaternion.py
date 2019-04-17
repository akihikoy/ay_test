#!/usr/bin/python
#import tf
import rostf
import math
import numpy as np
import numpy.linalg as la

def QFromAxisAngle(axis,angle):
  axis= axis / la.norm(axis)
  return rostf.quaternion_about_axis(angle,axis)

#Quaternion to 3x3 rotation matrix
def ROSQToRot(q):
  return rostf.quaternion_matrix(q)[:3,:3]

#3x3 rotation matrix to quaternion
def ROSRotToQ(R):
  M = rostf.identity_matrix()
  M[:3,:3] = R
  return rostf.quaternion_from_matrix(M)

#Quaternion to 3x3 rotation matrix
#cf. http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
def QToRot(q):
  R= np.array([[0.0]*3]*3)
  qx= q[0]
  qy= q[1]
  qz= q[2]
  qw= q[3]
  sqw = qw*qw
  sqx = qx*qx
  sqy = qy*qy
  sqz = qz*qz

  #invs (inverse square length) is only required if quaternion is not already normalised
  invs = 1.0 / (sqx + sqy + sqz + sqw)
  R[0,0] = ( sqx - sqy - sqz + sqw)*invs  #since sqw + sqx + sqy + sqz =1/invs*invs
  R[1,1] = (-sqx + sqy - sqz + sqw)*invs
  R[2,2] = (-sqx - sqy + sqz + sqw)*invs

  tmp1 = qx*qy
  tmp2 = qz*qw
  R[1,0] = 2.0 * (tmp1 + tmp2)*invs
  R[0,1] = 2.0 * (tmp1 - tmp2)*invs

  tmp1 = qx*qz
  tmp2 = qy*qw
  R[2,0] = 2.0 * (tmp1 - tmp2)*invs
  R[0,2] = 2.0 * (tmp1 + tmp2)*invs
  tmp1 = qy*qz
  tmp2 = qx*qw
  R[2,1] = 2.0 * (tmp1 + tmp2)*invs
  R[1,2] = 2.0 * (tmp1 - tmp2)*invs
  return R

#3x3 rotation matrix to quaternion
#cf. http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
def RotToQ(R):
  q= [0.0]*4
  trace= R[0,0] + R[1,1] + R[2,2]
  if trace>0.0:
    s = 0.5 / math.sqrt(trace+1.0)
    q[0] = ( R[2,1] - R[1,2] ) * s
    q[1] = ( R[0,2] - R[2,0] ) * s
    q[2] = ( R[1,0] - R[0,1] ) * s
    q[3] = 0.25 / s
  else:
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
      s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
      q[0] = 0.25 * s
      q[1] = (R[0,1] + R[1,0] ) / s
      q[2] = (R[0,2] + R[2,0] ) / s
      q[3] = (R[2,1] - R[1,2] ) / s
    elif R[1,1] > R[2,2]:
      s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
      q[0] = (R[0,1] + R[1,0] ) / s
      q[1] = 0.25 * s
      q[2] = (R[1,2] + R[2,1] ) / s
      q[3] = (R[0,2] - R[2,0] ) / s
    else:
      s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
      q[0] = (R[0,2] + R[2,0] ) / s
      q[1] = (R[1,2] + R[2,1] ) / s
      q[2] = 0.25 * s
      q[3] = (R[1,0] - R[0,1] ) / s
  return q

if __name__=='__main__':
  import random
  rand1= lambda: 2.0*(random.random()-0.5)
  q= QFromAxisAngle([rand1() for d in range(3)], math.pi*rand1())
  #q= [0.05240207810290061, -0.620853105352924, -0.6535491923757895, 0.4297311914779772]
  q= np.array(q)
  print 'q=',q
  print 'la.norm(q)=',la.norm(q)
  print 'ROSQToRot(q)=',ROSQToRot(q)
  print 'ROSQToRot(-q)=',ROSQToRot(-q)
  print 'ROSRotToQ(ROSQToRot(q))=',ROSRotToQ(ROSQToRot(q))
  print 'QToRot(q)=',QToRot(q)
  print 'QToRot(-q)=',QToRot(-q)
  print 'RotToQ(QToRot(q))=',RotToQ(QToRot(q))

  print 'ROSQToRot(q)==QToRot(q)?', np.allclose(ROSQToRot(q),QToRot(q))
  print 'ROSQToRot(-q)==QToRot(-q)?', np.allclose(ROSQToRot(-q),QToRot(-q))
  print 'ROSRotToQ(ROSQToRot(q))==RotToQ(QToRot(q))?', np.allclose(ROSRotToQ(ROSQToRot(q)),RotToQ(QToRot(q)))
