#!/usr/bin/python3
from polygon_point_in_out import *
import math
import numpy as np
import copy

def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

#x,y wave pattern generator
class TWaveGenerator:
  def __init__(self,vx=0.1):
    self.KeyPts= [[0.0 ,vx*0.0 , 0.0],
                  [0.25,vx*0.25, 1.0],
                  [0.75,vx*0.75,-1.0],
                  [1.0 ,vx*1.0 , 0.0]]
    self.idx_prev= 0

  def FindIdx(self, t, idx_prev=0):
    idx= idx_prev
    if idx>=len(self.KeyPts): idx= len(self.KeyPts)-1
    while idx+1<len(self.KeyPts) and t>self.KeyPts[idx+1][0]:  idx+=1
    while idx>=0 and t<self.KeyPts[idx][0]:  idx-=1
    return idx

  def FindKeyPts(self, t, m1=1.0, m2=1.0):
    idx= self.FindIdx(t,self.idx_prev)
    if idx<0 or idx>=len(self.KeyPts)-1:
      print('WARNING: Given t= %f is out of the key points (index: %i)' % (t,idx))
      if idx<0:
        idx= 0
        t= self.KeyPts[0].T
      else:
        idx= len(self.KeyPts)-2
        t= self.KeyPts[-1].T
    self.idx_prev= idx
    p0= copy.deepcopy(self.KeyPts[idx])
    p1= copy.deepcopy(self.KeyPts[idx+1])
    if idx==1:    p0[2]*= m1
    if idx+1==1:  p1[2]*= m1
    if idx==2:    p0[2]*= m2
    if idx+1==2:  p1[2]*= m2
    return p0,p1

  def Evaluate(self, t, m1=1.0, m2=1.0):
    p0,p1= self.FindKeyPts(t, m1, m2)
    return [ (p1[1]-p0[1])/(p1[0]-p0[0])*(t-p0[0])+p0[1],
             (p1[2]-p0[2])/(p1[0]-p0[0])*(t-p0[0])+p0[2]]

#def Repeater(base, t):
  #ti= Mod(t,1.0)
  #n= (t-ti)/1.0
  #start= np.array(base(1.0))*n
  #return start + np.array(base(ti))

def IdSampler(N):
  if N==0:  return []
  if N==1:  return [0]
  if N==2:  return [0,1]
  src= list(range(N))
  res= []
  res.append(src.pop(0))
  res.append(src.pop(-1))
  d= 2
  while True:
    for i in range(1,d,2):
      res.append(src.pop(len(src)*i//d))
      if len(src)==0:  return res
    d*= 2

def FSampler(xmin,xmax,num_div):
  data= FRange(xmin,xmax,num_div)
  return [data[i] for i in IdSampler(int(num_div))]

#p= func(t)
def EvalWaveFunc(func, points, resolution=20):
  e1= True
  e2= True
  for t in FSampler(0.0,0.5,resolution/2):
    p= func(t)
    if not PointInPolygon2D(points,p):
      e1= False
      break
  for t in FSampler(0.5,1.0,resolution/2):
    p= func(t)
    if not PointInPolygon2D(points,p):
      e2= False
      break
  return e1, e2

#p= func(t,p1,p2)
#True/False,True/False= eval_func(lambda t:func(t,p1,p2))
def OptimizeWaveFunc(func, p1_0, p2_0, eval_func):
  #Check the existence of the solution:
  e1,e2= eval_func(lambda t:func(t,0.0,0.0))
  if not e1 or not e2:  return None,None

  p1= p1_0
  p2= p2_0
  while True:
    e1,e2= eval_func(lambda t:func(t,p1,p2))
    #print p1,p2,e1,e2
    if e1 and e2:  return p1,p2
    if not e1:
      p1*= 0.9
      if p1<1.0e-6:
        p2*= 0.9
    if not e2:
      p2*= 0.9
      if p2<1.0e-6:
        p1*= 0.9

if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))

  from gen_data import *
  #points= To2d(Gen3d_01())
  #points= To2d2(Gen3d_02())
  #points= To2d2(Gen3d_11())
  points= To2d2(Gen3d_12())
  #points= To2d2(Gen3d_13())

  fp= open('/tmp/orig.dat','w')
  for p in points.tolist()+[points[0].tolist()]:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  pca= TPCA(points)

  u_dir= pca.EVecs[0]
  print('direction=',u_dir)
  direction= math.atan2(u_dir[1],u_dir[0])
  start= pca.Mean
  while True:
    start2= np.array(start)-0.01*u_dir
    if not PointInPolygon2D(points, start2):  break
    start= start2
  print('start=',start)
  rot= np.array([[math.cos(direction),-math.sin(direction)],[math.sin(direction),math.cos(direction)]])

  wave= TWaveGenerator()

  ##Test spreading wave (without planning)
  #fp= open('/tmp/spread1.dat','w')
  #n_old= 0
  #for t in FRange(0.0,10.0,120):
    #ti= Mod(t,1.0)
    #n= (t-ti)/1.0
    #if n!=n_old:
      #start= np.array(start) + np.dot(rot, np.array(wave.Evaluate(1.0)))
      #n_old= n
    #p= np.array(start) + np.dot(rot, np.array(wave.Evaluate(ti)))
    #fp.write(' '.join(map(str,p))+'\n')
  #fp.close()

  #Spreading wave (with planning)
  fp= open('/tmp/spread2.dat','w')
  while True:
    func= lambda ti,p1,p2: np.array(start) + np.dot(rot, np.array(wave.Evaluate(ti,m1=p1,m2=p2)))
    p1o,p2o= OptimizeWaveFunc(func, p1_0=2.0, p2_0=2.0, eval_func=lambda f:EvalWaveFunc(f,points))
    if None in (p1o,p2o):  break
    print(p1o, p2o, EvalWaveFunc(lambda t:func(t,p1o,p2o),points), func(0.0,p1o,p2o), PointInPolygon2D(points,func(0.0,p1o,p2o)))
    for t in FRange(0.0,1.0,20):
      p= func(t,p1o,p2o)
      fp.write(' '.join(map(str,p))+'\n')
    start= p
  fp.close()

  print('Plot by')
  print("qplot -x -s 'set size ratio -1' /tmp/orig.dat w l /tmp/spread2.dat w l")

