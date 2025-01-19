#!/usr/bin/python3
import numpy as np
import numpy.linalg as la
import math

def GetWedge(w):
  wedge= np.zeros((3,3))
  wedge[0,0]=0.0;    wedge[0,1]=-w[2];  wedge[0,2]=w[1]
  wedge[1,0]=w[2];   wedge[1,1]=0.0;    wedge[1,2]=-w[0]
  wedge[2,0]=-w[1];  wedge[2,1]=w[0];   wedge[2,2]=0.0
  return wedge

def Rodrigues(w, epsilon=1.0e-6):
  th= la.norm(w)
  if th<epsilon:  return np.identity(3)
  w_wedge= GetWedge(w *(1.0/th))
  return np.identity(3) + w_wedge * math.sin(th) + np.dot(w_wedge,w_wedge) * (1.0-math.cos(th))

#Inverse of Rodrigues, i.e. returns w (=angle*axis) from R where angle is in radian and axis is 3D unit vector.
def InvRodrigues0(R, epsilon=1.0e-6):
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

#Inverse of Rodrigues, i.e. returns w (=angle*axis) from R where angle is in radian and axis is 3D unit vector.
#With singularity detection. Src: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
def InvRodrigues(R, epsilon=1.0e-6):
  epsilon2= epsilon
  if abs(R[0,1]-R[1,0])<epsilon and abs(R[0,2]-R[2,0])<epsilon and abs(R[1,2]-R[2,1])<epsilon:
    #singularity found
    #first check for identity matrix which must have +1 for all terms
    #in leading diagonaland zero in other terms
    if abs(R[0,1]+R[1,0])<epsilon2 and abs(R[0,2]+R[2,0])<epsilon2 and abs(R[1,2]+R[2,1])<epsilon2 and abs(R[0,0]+R[1,1]+R[2,2]-3)<epsilon2:
      #this singularity is identity matrix so angle = 0
      return np.zeros(3)
    #otherwise this singularity is angle = 180
    angle= np.pi
    xx= (R[0,0]+1.)/2.
    yy= (R[1,1]+1.)/2.
    zz= (R[2,2]+1.)/2.
    xy= (R[0,1]+R[1,0])/4.
    xz= (R[0,2]+R[2,0])/4.
    yz= (R[1,2]+R[2,1])/4.
    if xx>yy and xx>zz:
      #R[0,0] is the largest diagonal term
      if xx<epsilon:
        x,y,z= 0, np.cos(np.pi/4.), np.cos(np.pi/4.)
      else:
        x= np.sqrt(xx)
        y,z= xy/x, xz/x
    elif yy > zz:
      #R[1,1] is the largest diagonal term
      if yy<epsilon:
        x,y,z= np.cos(np.pi/4.), 0.0, np.cos(np.pi/4.)
      else:
        y= np.sqrt(yy)
        x,z= xy/y, yz/y
    else:
      #R[2,2] is the largest diagonal term so base result on this
      if zz<epsilon:
        x,y,z= np.cos(np.pi/4.), np.cos(np.pi/4.), 0.0
      else:
        z= np.sqrt(zz)
        x,y= xz/z, yz/z
    return angle*np.array([x,y,z])
  #as we have reached here there are no singularities so we can handle normally
  s= np.sqrt((R[2,1]-R[1,2])*(R[2,1]-R[1,2])+(R[0,2]-R[2,0])*(R[0,2]-R[2,0])+(R[1,0]-R[0,1])*(R[1,0]-R[0,1]))
  if np.abs(s)<epsilon:  s=1.0
  #prevent divide by zero, should not happen if matrix is orthogonal and should be
  #caught by singularity test above, but I've left it in just in case
  angle= np.arccos((R[0,0]+R[1,1]+R[2,2]-1.)/2.)
  tmp= angle/s
  return tmp*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

#Quaternion from rotation matrix
def QuaternionFromMatrix(matrix):
  q = np.empty((4, ), dtype=np.float64)
  M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
  M33 = 1.0
  t = np.trace(M)+M33
  if t > M33:
    q[3] = t
    q[2] = M[1, 0] - M[0, 1]
    q[1] = M[0, 2] - M[2, 0]
    q[0] = M[2, 1] - M[1, 2]
  else:
    i, j, k = 0, 1, 2
    if M[1, 1] > M[0, 0]:
        i, j, k = 1, 2, 0
    if M[2, 2] > M[i, i]:
        i, j, k = 2, 0, 1
    t = M[i, i] - (M[j, j] + M[k, k]) + M33
    q[i] = t
    q[j] = M[i, j] + M[j, i]
    q[k] = M[k, i] + M[i, k]
    q[3] = M[k, j] - M[j, k]
  q *= 0.5 / math.sqrt(t * M33)
  return q

if __name__=='__main__':
  fp= open('/tmp/quaternion.dat','w')
  R1= Rodrigues(np.array([0.1,-0.5,-0.5]))
  R2= Rodrigues(np.array([1.0,0.5,0.5]))
  print('R1=',R1)
  fp.write(' '.join(map(str,QuaternionFromMatrix(R1)))+'\n')
  print('------------')
  #print np.dot(R1.T,R1)
  #print np.dot(R2.T,R2)
  trans_R= np.dot(R2,R1.T)
  #print np.dot(trans_R,R1)
  w= InvRodrigues(trans_R)
  N= 10
  for t in range(N):
    R= np.dot(Rodrigues(float(t+1)/float(N)*w),R1)
    print(R)
    fp.write(' '.join(map(str,QuaternionFromMatrix(R)))+'\n')
  print('------------')
  print('R2=',R2)
  fp.write(' '.join(map(str,QuaternionFromMatrix(R2)))+'\n')

