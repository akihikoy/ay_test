#! /usr/bin/env python
#Basic tools (geometry).
import numpy as np
import numpy.linalg as la
import math
from util import *
import _rostf

#Convert radian to [-pi,pi)
def AngleMod1(q):
  return Mod(q+math.pi,math.pi*2.0)-math.pi

#Convert radian to [0,2*pi)
def AngleMod2(q):
  return Mod(q,math.pi*2.0)

def DegToRad(q):
  conv= lambda x: float(x)/180.0*math.pi
  if type(q) in (float,int):
    return conv(q)
  else:
    return map(conv, q)

def RadToDeg(q):
  conv= lambda x: float(x)/math.pi*180.0
  if type(q) in (float,int):
    return conv(q)
  else:
    return map(conv, q)

#Displacement of two angles (angle2-angle1), whose absolute value is less than pi
def AngleDisplacement(angle1, angle2):
  angle1= AngleMod1(angle1)
  angle2= AngleMod1(angle2)
  if angle2>=angle1:
    d= angle2-angle1
    return d if d<=math.pi else d-2.0*math.pi
  else:
    d= angle1-angle2
    return -d if d<=math.pi else 2.0*math.pi-d

#Check if an angle is between [a_range[0],a_range[1]]
def IsAngleIn(angle, a_range):
  a_range= map(AngleMod1,a_range)
  if a_range[0]<a_range[1]:
    if a_range[1]-a_range[0]>math.pi:  return angle<=a_range[0] or  a_range[1]<=angle
    else:                              return a_range[0]<=angle and angle<=a_range[1]
  else:
    if a_range[0]-a_range[1]>math.pi:  return angle<=a_range[1] or  a_range[0]<=angle
    else:                              return a_range[1]<=angle and angle<=a_range[0]

#Calculating an angle [0,pi] between two 3-D vectors
def GetAngle(p1,p2):
  cos_th= np.dot(p1,p2) / (la.norm(p1)*la.norm(p2))
  if cos_th>1.0:  cos_th=1.0
  elif cos_th<-1.0:  cos_th=-1.0
  return math.acos(cos_th)

#Return axis,angle with which p1 is rotated to p2's direction
def GetAxisAngle(p1,p2):
  p1xp2= np.cross(p1,p2)
  p1xp2_norm= Norm(p1xp2)
  if p1xp2_norm<1.0e-8:  return [1.0,0.0,0.0],0.0
  axis= p1xp2/p1xp2_norm
  ex= Normalize(p1)
  ey= np.cross(axis,ex)
  angle= math.atan2(np.dot(ey,p2), np.dot(ex,p2))
  return axis,angle

#Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
#original_norm: keep original vec's norm, otherwise the norm is 1.
#Using The Gram-Schmidt process: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def Orthogonalize(vec, base, original_norm=True):
  base= np.array(base)/la.norm(base)
  vec2= vec - np.dot(vec,base)*base
  if original_norm:  return vec2 / la.norm(vec2) * la.norm(vec)
  else:              return vec2 / la.norm(vec2)

#Get an orthogonal axis of a given axis
#preferable: preferable axis (orthogonal axis is close to this)
#fault: return this axis when dot(axis,preferable)==1
def GetOrthogonalAxisOf(axis,preferable=[0.0,0.0,1.0],fault=None):
  axis= np.array(axis)/la.norm(axis)
  if fault is None or 1.0-abs(np.dot(axis,preferable))>=1.0e-6:
    return Orthogonalize(preferable,base=axis,original_norm=False)
  else:
    return fault

def QFromAxisAngle(axis,angle):
  axis= axis / la.norm(axis)
  return _rostf.quaternion_about_axis(angle,axis)

def RFromAxisAngle(axis,angle):
  return QToRot(QFromAxisAngle(axis,angle))

#Quaternion to 3x3 rotation matrix
def QToRot(q):
  return _rostf.quaternion_matrix(q)[:3,:3]

#3x3 rotation matrix to quaternion
def RotToQ(R):
  M = _rostf.identity_matrix()
  M[:3,:3] = R
  return _rostf.quaternion_from_matrix(M)

#Convert a pose, x,y,z,quaternion(qx,qy,qz,qw) to pos (x,y,z) and 3x3 rotation matrix
def XToPosRot(x):
  p = np.array(x[0:3])
  R = _rostf.quaternion_matrix(x[3:7])[:3,:3]
  return p, R

#Convert pos p=(x,y,z) and 3x3 rotation matrix R to a pose, x,y,z,quaternion(qx,qy,qz,qw)
def PosRotToX(p,R):
  M = _rostf.identity_matrix()
  M[:3,:3] = R
  x = list(p)+[0.0]*4
  x[3:7] = _rostf.quaternion_from_matrix(M)
  return x

#Decompose a rotation matrix R to axis vectors ex,ey,ez
def RotToExyz(R):
  #ex,ey,ez
  return R[:,0], R[:,1], R[:,2]

#Compose axis vectors ex,ey,ez to make a rotation matrix R
def ExyzToRot(ex,ey,ez):
  R= np.zeros((3,3))
  R[:,0]= ex
  R[:,1]= ey
  R[:,2]= ez
  return R

def GetWedge(w):
  wedge= np.zeros((3,3))
  wedge[0,0]=0.0;    wedge[0,1]=-w[2];  wedge[0,2]=w[1]
  wedge[1,0]=w[2];   wedge[1,1]=0.0;    wedge[1,2]=-w[0]
  wedge[2,0]=-w[1];  wedge[2,1]=w[0];   wedge[2,2]=0.0
  return wedge

def Rodrigues(w, epsilon=1.0e-6):
  th= la.norm(w)
  if th<epsilon:  return np.identity(3)
  w_wedge= GetWedge(np.array(w) *(1.0/th))
  return np.identity(3) + w_wedge * math.sin(th) + np.dot(w_wedge,w_wedge) * (1.0-math.cos(th))

def InvRodrigues(R, epsilon=1.0e-6):
  alpha= (R[0,0]+R[1,1]+R[2,2] - 1.0) / 2.0

  if (alpha-1.0 < epsilon) and (alpha-1.0 > -epsilon):
    return np.array([0.0,0.0,0.0])
  else:
    w= np.zeros(3)
    th = math.acos(alpha)
    tmp= 0.5 * th / math.sin(th)
    w[0] = tmp * (R[2,1] - R[1,2])
    w[1] = tmp * (R[0,2] - R[2,0])
    w[2] = tmp * (R[1,0] - R[0,1])
    return w

#Multiply two quaternions
def MultiplyQ(q1,q2):
  return _rostf.quaternion_multiply(q1,q2)

#This solves for x in "x_r = x_l * x", i.e. return "inv(x_l)*x_r"
#For example, get a local pose of x_r in the x_l frame
#x_* are [x,y,z,quaternion] form
#x_r can be also [x,y,z]
def TransformLeftInv(x_l,x_r):
  if len(x_r)==3:
    pl,Rl= XToPosRot(x_l)
    pr= x_r
    p= np.dot(Rl.T, (pr-pl))
    return p
  elif len(x_r)==7:
    pl,Rl= XToPosRot(x_l)
    pr,Rr= XToPosRot(x_r)
    p= np.dot(Rl.T, (pr-pl))
    R= np.dot(Rl.T, Rr)
    return PosRotToX(p,R)

#This solves for trans_x in "x_l = trans_x * x_r", i.e. return "x_l*inv(x_r)"
#For example, get a transformation, x_r to x_l
#x_* are [x,y,z,quaternion] form
def TransformRightInv(x_l,x_r):
  pl,Rl= XToPosRot(x_l)
  pr,Rr= XToPosRot(x_r)
  Rt= np.dot(Rl, Rr.T)
  pt= pl-np.dot(Rt,pr)
  return PosRotToX(pt,Rt)

#Compute "x2 * x1"; x* are [x,y,z,quaternion] form
#x1 can also be [x,y,z] or [quaternion]
#x2 can also be [x,y,z] or [quaternion]
def Transform(x2, x1):
  if len(x2)==3:
    if len(x1)==7:
      x3= [0.0]*7
      x3[:3]= Vec(x2)+Vec(x1[:3])
      x3[3:]= x1[3:]
      return x3
    if len(x1)==3:  #i.e. [x,y,z]
      return Vec(x2)+Vec(x1)
    if len(x1)==4:  #i.e. [quaternion]
      raise Exception('invalid Transform: point * quaternion')

  if len(x2)==7:
    p2,R2= XToPosRot(x2)
  elif len(x2)==4:  #i.e. [quaternion]
    p2= Vec([0.0,0.0,0.0])
    R2= QToRot(x2)

  if len(x1)==7:
    p1,R1= XToPosRot(x1)
    p= np.dot(R2,p1)+p2
    R= np.dot(R2, R1)
    return PosRotToX(p,R)
  if len(x1)==3:  #i.e. [x,y,z]
    p1= x1
    p= np.dot(R2,p1)+p2
    return p
  if len(x1)==4:  #i.e. [quaternion]
    R1= QToRot(x1)
    R= np.dot(R2, R1)
    return RotToQ(R)

#Get weighted average of two rotation matrices (intuitively, (1-w2)*R1 + w2*R2)
def AverageRot(R1, R2, w2):
  w= InvRodrigues(np.dot(R2,R1.T))
  return np.dot(Rodrigues(w2*w),R1)

#Get weighted average of two rotation quaternions (intuitively, (1-w2)*q1 + w2*q2)
def AverageQ(q1, q2, w2):
  return RotToQ( AverageRot(QToRot(q1), QToRot(q2), w2) )

#Get weighted average of two poses (intuitively, (1-w2)*x1 + w2*x2)
def AverageX(x1, x2, w2):
  x= [0]*7
  x[0:3]= (1.0-w2)*np.array(x1[0:3])+w2*np.array(x2[0:3])
  x[3:]= AverageQ(x1[3:], x2[3:], w2)
  return x

#Get difference of two poses [dx,dy,dz, dwx,dwy,dwz] (like x2-x1)
def DiffX(x1, x2):
  w= InvRodrigues(np.dot(QToRot(x2[3:]),QToRot(x1[3:]).T))
  return [x2[0]-x1[0],x2[1]-x1[1],x2[2]-x1[2], w[0],w[1],w[2]]

#Add a difference of pose dx to a pose x
#where dx is like a result of DiffX
def AddDiffX(x1, dx):
  x= [0]*7
  x[:3]= [x1[0]+dx[0], x1[1]+dx[1], x1[2]+dx[2]]
  x[3:]= RotToQ( np.dot(Rodrigues(dx[3:]),QToRot(x1[3:])) )
  return x

#Get average of poses
def AverageXData(x_data):
  if len(x_data)==0:  return []
  x_avr= x_data[0]
  for i in range(1,len(x_data)):
    w2= 1.0/(1.0+float(i))
    x_avr= AverageX(x_avr, x_data[i], w2)
  return x_avr

#For visualizing cylinder, arrow, etc., get a pose x from two points p1-->p2.
#Axis ax decides which axis corresponds to p1-->p2.
#Ratio r decides: r=0: x is on p1, r=1: x is on p2, r=0.5: x is on the middle of p1 and p2.
def XFromP1P2(p1, p2, ax='z', r=0.5):
  if ax=='x':
    ex= Normalize(Vec(p2)-Vec(p1))
    ey= GetOrthogonalAxisOf(ex,preferable=[0.0,1.0,0.0],fault=[0.0,0.0,1.0])
    ez= np.cross(ex,ey)
  elif ax=='y':
    ey= Normalize(Vec(p2)-Vec(p1))
    ez= GetOrthogonalAxisOf(ey,preferable=[0.0,0.0,1.0],fault=[1.0,0.0,0.0])
    ex= np.cross(ey,ez)
  elif ax=='z':
    ez= Normalize(Vec(p2)-Vec(p1))
    ex= GetOrthogonalAxisOf(ez,preferable=[1.0,0.0,0.0],fault=[0.0,1.0,0.0])
    ey= np.cross(ez,ex)
  x= [0]*7
  x[0:3]= (1.0-r)*Vec(p1)+r*Vec(p2)
  x[3:]= RotToQ(np.matrix([ex,ey,ez]).T)
  return x


'''Project 3D point to an image plane.
We assume a project matrix:
        [[Fx  0  Cx]
    P =  [ 0  Fy Cy]
         [ 0  0   1]]
Camera is at [0,0,0] and the image plane is z=1.
  A 3D point [xc,yc,zc]^T is projected onto an image plane [xp,yp] by:
    [u,v,w]^T= P * [xc,yc,zc]^T
    xp= u/w
    yp= v/w '''
def ProjectPointToImage(pt3d, P, zmin=0.001):
  if pt3d[2]<zmin:  return None
  pt2d= np.dot(P[:3,:3], pt3d)
  return [pt2d[0]/pt2d[2], pt2d[1]/pt2d[2]]

#Inverse project a point pt2d=[xp,yp] on image to 3D point pt3d=[xc,yc,1]
def InvProjectFromImage(pt2d, P):
  Fx,Fy,Cx,Cy= P[0,0],P[1,1],P[0,2],P[1,2]
  return [(pt2d[0]-Cx)/Fx, (pt2d[1]-Cy)/Fy, 1.0]

