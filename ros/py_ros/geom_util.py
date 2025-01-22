#!/usr/bin/python3
#\file    geom_util.py
#\brief   Geometry utility (using ROS tf).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.31, 2015
import numpy as np
import numpy.linalg as la
import tf
import copy
import math
import random
import geometry_msgs.msg

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

#Float version of range
def FRange1(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

def Vec(x):
  return np.array(x)

def Mat(x):
  return np.mat(x)

def Eye(N=3):
  return np.eye(N)

def NormSq(x):
  #s= 0.0
  #for xd in x:  s+= xd**2
  #return s
  return la.norm(x)**2

#L2 norm of a vector x
def Norm(x):
  return la.norm(x)

#Max norm of a vector x
def MaxNorm(x):
  return max(list(map(abs,x)))


#Return a normalized vector with L2 norm
def Normalize(x):
  return np.array(x)/la.norm(x)

#Distance of two vectors: L2 norm of their difference
def Dist(p1,p2):
  return la.norm(np.array(p2)-p1)

#Distance of two vectors: Max norm of their difference
def DistM(p1,p2):
  return np.abs(np.array(p2)-p1).max()



#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

#Generate a random vector of uniform distribution; each dim has the same bound.
def RandVec(nd,xmin=-0.5,xmax=0.5):
  return Vec([Rand(xmin,xmax) for d in range(nd)])

#Generate a random vector of uniform distribution; each dim has different bound.
def RandN(xmins,xmaxs):
  assert(len(xmins)==len(xmaxs))
  return [Rand(xmins[d],xmaxs[d]) for d in range(len(xmins))]

#Generate a random vector of uniform distribution; each dim has different bound.
def RandB(bounds):
  return RandN(bounds[0],bounds[1])




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
    return list(map(conv, q))

def RadToDeg(q):
  conv= lambda x: float(x)/math.pi*180.0
  if type(q) in (float,int):
    return conv(q)
  else:
    return list(map(conv, q))

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
  a_range= list(map(AngleMod1,a_range))
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
  axis= Normalize(np.cross(p1,p2))
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
  return tf.transformations.quaternion_about_axis(angle,axis)

def RFromAxisAngle(axis,angle):
  return QToRot(QFromAxisAngle(axis,angle))

#Quaternion to 3x3 rotation matrix
def QToRot(q):
  return tf.transformations.quaternion_matrix(q)[:3,:3]

#3x3 rotation matrix to quaternion
def RotToQ(R):
  M = tf.transformations.identity_matrix()
  M[:3,:3] = R
  return tf.transformations.quaternion_from_matrix(M)

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

#Convert p to geometry_msgs/Point
def PToGPoint(p):
  point= geometry_msgs.msg.Point()
  point.x= p[0]
  point.y= p[1]
  point.z= p[2]
  return point

#Convert x to geometry_msgs/Pose
def XToGPose(x):
  pose= geometry_msgs.msg.Pose()
  pose.position.x= x[0]
  pose.position.y= x[1]
  pose.position.z= x[2]
  pose.orientation.x= x[3]
  pose.orientation.y= x[4]
  pose.orientation.z= x[5]
  pose.orientation.w= x[6]
  return pose

#Convert geometry_msgs/Pose to x
def GPoseToX(pose):
  x= [0]*7
  x[0]= pose.position.x
  x[1]= pose.position.y
  x[2]= pose.position.z
  x[3]= pose.orientation.x
  x[4]= pose.orientation.y
  x[5]= pose.orientation.z
  x[6]= pose.orientation.w
  return x

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
  return tf.transformations.quaternion_multiply(q1,q2)

#This solves for x in "x_r = x_l * x", i.e. return "inv(x_l)*x_r"
#For example, get a local pose of x_r in the x_l frame
#x_* are [x,y,z,quaternion] form
def TransformLeftInv(x_l,x_r):
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





#Get a sequence of times, from 0 to dt including inum points (0 is not included).
def TimeTraj(dt, inum):
  return FRange1(0.0, dt, inum)[1:]

#Remove radian jumping in a joint angle trajectory.
def SmoothQTraj(q_traj):
  if len(q_traj)==0:  return
  q_prev= np.array(q_traj[0])
  q_offset= np.array([0]*len(q_prev))
  for q in q_traj:
    q_diff= np.array(q) - q_prev
    for d in range(len(q_prev)):
      if q_diff[d]<-math.pi:  q_offset[d]+=1
      elif q_diff[d]>math.pi:  q_offset[d]-=1
    q_prev= copy.deepcopy(q)
    q[:]= q+q_offset*2.0*math.pi

'''Limit joint velocities.
Sequence of times t_traj will be directly modified.
  q_start: joint angles at t=0.
  q_traj: joint angle trajectory [q0,...,qD]*N.
  t_traj: corresponding times in seconds from start [t1,t2,...,tN].
  qvel_limits: limit velocities. '''
def LimitQTrajVel(q_start, q_traj, t_traj, qvel_limits, termlen=9, dv=0.08):
  assert(len(q_traj)==len(t_traj))
  if len(q_traj)==0:  return

  def sub_proc(qt, i, t_offset, t_prev, q_prev, vel_limits):
    q= qt[0][i]
    qt[1][i]+= t_offset
    qd= [AngleMod1(q[d]-q_prev[d]) / (qt[1][i]-t_prev) for d in range(len(q))]
    s= max([abs(qd[d])/vel_limits[d] for d in range(len(qd))])
    if s>1.0:
      diff= (qt[1][i]-t_prev)*float(s-1.0)
      t_offset+= diff
      qt[1][i]+= diff
    CPrint(2,'s=',s,qt[1][i]-t_prev,qd)
    return t_offset, qt[1][i], q

  t_offset= 0.0
  t_prev= 0.0
  q_prev= q_start
  qt= (q_traj,t_traj)

  offset=0
  i_term= termlen if len(q_traj)>2*termlen else len(q_traj)/2
  i_middle= len(q_traj)-2*i_term
  for i in range(i_term):
    vel_limits= [v*dv*float(i+1+offset) for v in qvel_limits]
    t_offset,t_prev,q_prev= sub_proc(qt, i, t_offset, t_prev, q_prev, vel_limits)
  for i in range(i_middle):
    t_offset,t_prev,q_prev= sub_proc(qt, i_term+i, t_offset, t_prev, q_prev, qvel_limits)
  for i in range(i_term):
    vel_limits= [v*dv*float(i_term-i+offset) for v in qvel_limits]
    t_offset,t_prev,q_prev= sub_proc(qt, i_term+i_middle+i, t_offset, t_prev, q_prev, vel_limits)

#Return the interpolation from x1 to x2 with N points
#p1 is not included
def XInterpolation(x1,x2,N):
  p1,R1= XToPosRot(x1)
  p2,R2= XToPosRot(x2)
  dp= (p2-p1)/float(N)
  trans_R= np.dot(R2,R1.T)
  w= InvRodrigues(trans_R)
  traj=[]
  for t in range(N):
    R= np.dot(Rodrigues(float(t+1)/float(N)*w),R1)
    p1= p1+dp
    traj.append(PosRotToX(p1,R))
  return traj

#Return the interpolation from q1 to q2 with N points
#q1 is not included
def QInterpolation(q1,q2,N):
  R1= QToRot(q1)
  R2= QToRot(q2)
  trans_R= np.dot(R2,R1.T)
  w= InvRodrigues(trans_R)
  traj=[]
  for t in range(N):
    R= np.dot(Rodrigues(float(t+1)/float(N)*w),R1)
    traj.append(RotToQ(R))
  return traj

'''Transform a Cartesian trajectory to joint angle trajectory.
  func_ik: IK function (x, q_start).
  x_traj: pose sequence [x1, x2, ...].
  start_angles: joint angles used for initial pose of first IK.  '''
def XTrajToQTraj(func_ik, x_traj, start_angles):
  N= len(x_traj)
  q_prev= start_angles
  q_traj= None
  for x,n in zip(x_traj, list(range(N))):
    q= func_ik(x, q_prev)
    if q==None:  return None
    if q_traj==None:  q_traj= [[0.0]*len(q) for i in range(N)]
    q_traj[n][:]= q
    q_traj[n][:]= q
    q_prev= q
  SmoothQTraj(q_traj)
  return q_traj



