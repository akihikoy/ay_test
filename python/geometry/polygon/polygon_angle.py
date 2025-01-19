#!/usr/bin/python3
import math
import numpy as np
import numpy.linalg as la
from pca2 import TPCA

#Float version of range
def FRange1(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]

def Sign(x):
  if x==0.0:  return 0
  if x>0.0:   return 1
  if x<0.0:   return -1

#Check if a is between [a_range[0],a_range[1]]
def IsIn(a, a_range):
  if a_range[0]<a_range[1]:
    return a_range[0]<=a and a<=a_range[1]
  else:
    return a_range[1]<=a and a<=a_range[0]

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

#Displacement of two angles, whose absolute value is less than pi
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


#Centroid of a polygon
#ref. http://en.wikipedia.org/wiki/Centroid
def PolygonCentroid(points, pca_default=None, only_centroid=True):
  if len(points)==0:  return None
  if len(points)==1:  return points[0]
  assert(len(points[0])==3)
  if pca_default==None:
    pca= TPCA(points)
  else:
    pca= pca_default
  xy= pca.Projected[:,[0,1]]
  N= len(xy)
  xy= np.vstack((xy,[xy[0]]))  #Extend so that xy[N]==xy[0]
  A= 0.5*sum([xy[n,0]*xy[n+1,1] - xy[n+1,0]*xy[n,1] for n in range(N)])
  Cx= sum([(xy[n,0]+xy[n+1,0])*(xy[n,0]*xy[n+1,1]-xy[n+1,0]*xy[n,1]) for n in range(N)]) / (6.0*A)
  Cy= sum([(xy[n,1]+xy[n+1,1])*(xy[n,0]*xy[n+1,1]-xy[n+1,0]*xy[n,1]) for n in range(N)]) / (6.0*A)
  centroid= pca.Reconstruct([Cx,Cy],[0,1])
  if only_centroid:  return centroid
  else:  return centroid, [Cx,Cy]

#Get a parameterized polygon
class TParameterizedPolygon:
  def __init__(self, points):
    assert(len(points)>=3);
    pca= TPCA(points)
    self.Center, self.Center2d= PolygonCentroid(points, pca, only_centroid=False)
    self.PCAAxes= pca.EVecs
    self.PCAValues= pca.EVals
    self.PCAMean= pca.Mean
    angles= [0]*len(pca.Projected)
    dirc= 0
    for i in range(len(points)):
      diff= pca.Projected[i,[0,1]] - np.array(self.Center2d)
      angles[i]= math.atan2(diff[1],diff[0])
      if angles[i]>math.pi:  angles[i]-= 2.0*math.pi
      if i>0 and AngleDisplacement(angles[i-1],angles[i])>0: dirc+=1
      else: dirc-=1
    dirc= Sign(dirc)
    self.Angles= []
    self.Points= []
    self.Points2D= []
    aprev= angles[0]
    for i in range(len(points)):
      if i==0 or Sign(AngleDisplacement(aprev,angles[i]))==dirc:
        self.Angles.append(angles[i])
        self.Points.append(points[i])
        self.Points2D.append(pca.Projected[i,[0,1]])
        aprev= angles[i]
    self.Angles.append(angles[0])
    self.Points.append(points[0])
    self.Points2D.append(pca.Projected[0,[0,1]])
    self.IdxAngleMin= min(list(range(len(self.Angles))), key=lambda i: self.Angles[i])
    self.IdxAngleMax= max(list(range(len(self.Angles))), key=lambda i: self.Angles[i])
    self.EstimateAxes(dtheta=2.0*math.pi/50.0)

  def AngleToPoint(self, angle):
    angle= AngleMod1(angle)
    if angle<=self.Angles[self.IdxAngleMin]:
      i_closest= self.IdxAngleMin
      i_closest2= self.IdxAngleMax
      alpha= abs(angle-self.Angles[i_closest])
      alpha2= 2.0*math.pi-self.Angles[i_closest2]+angle
    elif angle>=self.Angles[self.IdxAngleMax]:
      i_closest= self.IdxAngleMax
      i_closest2= self.IdxAngleMin
      alpha= abs(angle-self.Angles[i_closest])
      alpha2= 2.0*math.pi+self.Angles[i_closest2]-angle
    else:
      i_closest= [i for i in range(len(self.Angles)-1) if IsAngleIn(angle,[self.Angles[i],self.Angles[i+1]])]
      i_closest= i_closest[0]
      i_closest2= i_closest+1
      alpha= abs(angle-self.Angles[i_closest])
      alpha2= abs(self.Angles[i_closest2]-self.Angles[i_closest]) - alpha
    if abs(alpha)<1.0e-6:  return np.array(self.Points[i_closest])
    if abs(alpha2)<1.0e-6:  return np.array(self.Points[i_closest2])
    pi= np.array(self.Points2D[i_closest])
    pi2= np.array(self.Points2D[i_closest2])
    c= self.Center2d
    ratio= la.norm(pi-c)/la.norm(pi2-c) * math.sin(alpha)/math.sin(alpha2)
    t= ratio/(1.0+ratio)
    assert(t>=0.0 and t<=1.0)
    return t*np.array(self.Points[i_closest2]) + (1.0-t)*np.array(self.Points[i_closest])

  def PointToAngle(self, point):
    diff= np.dot(point-self.PCAMean, self.PCAAxes)[[0,1]] - np.array(self.Center2d)
    return math.atan2(diff[1],diff[0])

  def EstimateAxes(self,dtheta):
    theta= 0.0
    points= []
    while theta<2.0*math.pi:
      p= self.AngleToPoint(theta)
      points.append(p)
      theta+= dtheta
    pca= TPCA(points)
    self.Axes= pca.EVecs
    self.AxValues= pca.EVals


from gen_data import *
#points= Gen3d_01()
#points= Gen3d_02()
points= Gen3d_11()
#points= Gen3d_12()
#points= Gen3d_13()

fp= open('/tmp/orig.dat','w')
for p in points:
  fp.write(' '.join(map(str,p))+'\n')
fp.close()


ppolygon= TParameterizedPolygon(points)
#print '\n'.join(map(str,ppolygon.Angles))
print('ppolygon.Center=',ppolygon.Center)
print('ppolygon.IdxAngleMin=',ppolygon.IdxAngleMin)
print('ppolygon.IdxAngleMax=',ppolygon.IdxAngleMax)
print('ppolygon.PCAAxes=',ppolygon.PCAAxes)
print('ppolygon.PCAValues=',ppolygon.PCAValues)
print('ppolygon.Axes=',ppolygon.Axes)
print('ppolygon.AxValues=',ppolygon.AxValues)

fp= open('/tmp/res.dat','w')
fp2= open('/tmp/resw.dat','w')
#for angle in FRange1(-math.pi*1.2, math.pi*1.2, 10):
for angle in FRange1(-math.pi, math.pi, 100):
  p= ppolygon.AngleToPoint(angle)
  if p is not None:
    a= ppolygon.PointToAngle(p)
    fp.write('%s %f %f\n' % (' '.join(map(str,ppolygon.Center)), 0.0, 0.0))
    fp.write('%s %f %f\n' % (' '.join(map(str,p)), angle, a))
    fp.write('\n\n')
    if abs(AngleDisplacement(a,angle))>0.001:
      print('######WARNING',angle,a,abs(AngleDisplacement(a,angle)))
      fp2.write('%s %f %f\n' % (' '.join(map(str,ppolygon.Center)), 0.0, 0.0))
      fp2.write('%s %f %f\n' % (' '.join(map(str,p)), angle, a))
      fp2.write('\n\n')
      #break
  else:
    print('Invalid angle',angle*180./math.pi)
fp.close()
fp2.close()


fp= open('/tmp/res2.dat','w')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.PCAAxes[:,0]))))
fp.write('\n\n')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.PCAAxes[:,1]))))
fp.write('\n\n')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.PCAAxes[:,2]))))
fp.write('\n\n')
fp.close()

fp= open('/tmp/res3.dat','w')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.Axes[:,0]))))
fp.write('\n\n')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.Axes[:,1]))))
fp.write('\n\n')
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center))))
fp.write('%s\n' % (' '.join(map(str,ppolygon.Center+0.1*ppolygon.Axes[:,2]))))
fp.write('\n\n')
fp.close()

print('Plot by')
print('polygon:')
print("qplot -x -3d /tmp/orig.dat w l /tmp/res.dat w l")
print('axes:')
print("qplot -x -3d /tmp/orig.dat w l /tmp/res.dat w l /tmp/res2.dat w l /tmp/res3.dat w l")

