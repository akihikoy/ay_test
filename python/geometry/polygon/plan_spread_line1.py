#!/usr/bin/python
from polygon_point_in_out import *
import math
import numpy as np

def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

def WaveFunc(t, vx=0.1, fx=1.0, fy=1.0):
  if t<0.0 or t>1.0:  return [0.0, 0.0]
  if t<0.25:  return [fx*vx*t, fy*(4.0*t)]
  if t<0.75:  return [fx*vx*t, fy*(-4.0*(t-0.25)+1.0)]
  return [fx*vx*t, fy*(4.0*(t-0.75)-1.0)]

#def Repeater(base, t):
  #ti= Mod(t,1.0)
  #n= (t-ti)/1.0
  #start= np.array(base(1.0))*n
  #return start + np.array(base(ti))

#p= func(t)
def EvalWaveFunc(func, points, resolution=20):
  for t in FRange(0.0,1.0,resolution):
    p= func(t)
    if not PointInPolygon2D(points,p):
      return False
  return True

#p= func(t,p1)
#True/False= eval_func(lambda t:func(t,p1))
def OptimizeWaveFunc(func, p1_0, eval_func):
  #Check the existence of the solution:
  if not eval_func(lambda t:func(t,0.0)):  return None
  p1= p1_0
  while True:
    e= eval_func(lambda t:func(t,p1))
    if e:  return p1
    p1*= 0.9

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  from gen_data import *
  #points= To2d(Gen3d_01())
  #points= To2d(Gen3d_02())
  #points= To2d(Gen3d_11())
  points= To2d(Gen3d_12())
  #points= To2d(Gen3d_13())

  fp= file('/tmp/orig.dat','w')
  for p in points.tolist()+[points[0].tolist()]:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  pca= TPCA(points)

  u_dir= pca.EVecs[0]
  print 'direction=',u_dir
  direction= math.atan2(u_dir[1],u_dir[0])
  start= [0.0, 0.0]  #TODO: ditto
  while True:
    start2= np.array(start)-0.01*u_dir
    if not PointInPolygon2D(points, start2):  break
    start= start2
  print 'start=',start
  rot= np.array([[math.cos(direction),-math.sin(direction)],[math.sin(direction),math.cos(direction)]])

  ##Test spreading wave (without planning)
  #fp= file('/tmp/spread1.dat','w')
  #n_old= 0
  #for t in FRange(0.0,10.0,120):
    #ti= Mod(t,1.0)
    #n= (t-ti)/1.0
    #if n!=n_old:
      #start= np.array(start) + np.dot(rot, np.array(WaveFunc(1.0)))
      #n_old= n
    #p= np.array(start) + np.dot(rot, np.array(WaveFunc(ti)))
    #fp.write(' '.join(map(str,p))+'\n')
  #fp.close()

  #Spreading wave (with planning)
  fp= file('/tmp/spread2.dat','w')
  while True:
    func= lambda ti,p1: np.array(start) + np.dot(rot, np.array(WaveFunc(ti,vx=0.1,fy=p1)))
    p1o= OptimizeWaveFunc(func, p1_0=2.0, eval_func=lambda f:EvalWaveFunc(f,points))
    if p1o is None:  break
    print p1o, EvalWaveFunc(lambda t:func(t,p1o),points), func(0.0,p1o), PointInPolygon2D(points,func(0.0,p1o))
    for t in FRange(0.0,1.0,20):
      p= func(t,p1o)
      fp.write(' '.join(map(str,p))+'\n')
    start= p
  fp.close()

  print 'Plot by'
  print "qplot -x -s 'set size ratio -1' /tmp/orig.dat w l /tmp/spread2.dat w l"

