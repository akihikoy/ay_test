#!/usr/bin/python
import sys
sys.path.append('..')
from splines.cubic_hermite_spline import TCubicHermiteSpline
from polygon_point_in_out import *
import math
import numpy as np
import copy

FRange1= FRange

#NOTE: This will decide the length of a single phase wave but actual duration
#  is decided through an actual execution where the velocity controller is applied
MAX_TIME= 1.0

#x,y wave pattern generator
class TWaveGenerator:
  def __init__(self,vx=0.1):
    self.data= [[0.0 *MAX_TIME,vx*0.0 , 0.0],
                [0.25*MAX_TIME,vx*0.25, 1.0],
                [0.75*MAX_TIME,vx*0.75,-1.0],
                [1.0 *MAX_TIME,vx*1.0 , 0.0]]
    #FINITE_DIFF, CARDINAL
    self.splines= [TCubicHermiteSpline() for d in range(len(self.data[0])-1)]
    for d in range(len(self.splines)):
      data_d= [[x[0],x[d+1]] for x in self.data]
      self.splines[d].Initialize(data_d, tan_method=self.splines[d].CARDINAL, end_tan=self.splines[d].CYCLIC, c=0.0, m=0.0)
    self.params= []  #Set of [m1,m2] used in Evaluate

  def Evaluate(self, t, m1=None, m2=None):
    if None in (m1,m2):
      n,tp= self.splines[0].PhaseInfo(t)
      n= int(n)
      if m1==None:  m1= self.params[n][0] if n<len(self.params) else 1.0
      if m2==None:  m2= self.params[n][1] if n<len(self.params) else 1.0
    self.splines[1].KeyPts[1].X= self.data[1][2] * m1
    self.splines[1].KeyPts[2].X= self.data[2][2] * m2
    self.splines[1].Update()
    return [self.splines[d].EvaluateC(t) for d in range(len(self.splines))]

#Index sampler good for searching
def IdSampler(N):
  if N==0:  return []
  if N==1:  return [0]
  if N==2:  return [0,1]
  src= range(N)
  res= []
  res.append(src.pop(0))
  res.append(src.pop(-1))
  d= 2
  while True:
    for i in range(1,d,2):
      res.append(src.pop(len(src)*i/d))
      if len(src)==0:  return res
    d*= 2

def FSampler(xmin,xmax,num_div):
  data= FRange1(xmin,xmax,num_div)
  return [data[i] for i in IdSampler(num_div)]

#Return the evaluation e1,e2 of trajectory p=func(t), t in [0,MAX_TIME].
#e1= whether the first half is inside the polygon.
#e2= whether the last half is inside the polygon.
def EvalWaveFunc(func, points, resolution=20):
  e1= True
  e2= True
  for t in FSampler(0.0*MAX_TIME,0.5*MAX_TIME,resolution/2):
    p= func(t)
    if not PointInPolygon2D(points,p):
      e1= False
      break
  for t in FSampler(0.5*MAX_TIME,1.0*MAX_TIME,resolution/2):
    p= func(t)
    if not PointInPolygon2D(points,p):
      e2= False
      break
  return e1, e2

#Optimize a single phase of wave trajectory p= func(t,p1,p2), t in [0,MAX_TIME],
#parameterized by p1,p2,
#so that p1,p2 satisfies the evaluation function eval_func.
#True/False,True/False= eval_func(lambda t:func(t,p1,p2))
#Greater p1,p2 are better.
#Optimization starts from p1_0, p2_0 that are the maximum values of the parameters.
#search_factor must be in (0.0,1.0); 0.99:slow,accurate, 0.01:fast,inaccurate.
def OptimizeWaveFunc1(func, p1_0, p2_0, eval_func, search_factor=0.9):
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
      p1*= search_factor
      if p1<1.0e-6:
        p2*= search_factor
    if not e2:
      p2*= search_factor
      if p2<1.0e-6:
        p1*= search_factor


import cma_es.cma as cma
#TEST
#Optimize spline parameters using CMA-ES
#p= func(t,p1,p2)
#True/False,True/False= eval_func(lambda t:func(t,p1,p2))
def OptimizeWaveFunc2(func, p1_0, p2_0, eval_func):
  #Check the existence of the solution:
  e1,e2= eval_func(lambda t:func(t,0.0,0.0))
  if not e1 or not e2:  return None,None

  to_fmin= lambda p,e:  (-2.0*p[0]**2-2.0*p[1]**2) if e[0] and e[1] else None
  fobj= lambda p: to_fmin( p, eval_func(lambda t:func(t,p[0],p[1])) )
  options = {'CMA_diagonal':1, 'verb_time':0}
  options['bounds']= [[0.0,0.0],[2.0,2.0]]
  options['tolfun']= 3.0e-1 # 1.0e-4
  options['verb_log']= False
  options['scaling_of_variables']= np.array([1.0,1.0])
  scale0= 1.0
  parameters0= [0.0, 0.0]
  #res= cma.fmin(fobj, parameters0, scale0, options)
  es= cma.CMAEvolutionStrategy(parameters0, scale0, options)
  solutions, scores= [], []
  count= 0
  while not es.stop():
    while len(solutions) < es.popsize:
      x= es.ask(1)[0]
      f= fobj(x)
      if f is not None:
        solutions.append(x)
        scores.append(f)
      #print x,f
    es.tell(solutions, scores)
    es.disp()
    #print 'es.result()@%i:'%(count),es.result()
    count+=1
    solutions, scores= [], []
  res= es.result()

  return res[0][0], res[0][1]


if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  from gen_data import *
  #points= To2d(Gen3d_01())*10.0
  #points= To2d2(Gen3d_02())
  #points= To2d2(Gen3d_11())
  points= To2d2(Gen3d_12())*0.5
  #points= To2d2(Gen3d_13())

  fp= file('/tmp/orig.dat','w')
  for p in points.tolist()+[points[0].tolist()]:
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  pca= TPCA(points)

  u_dir= pca.EVecs[0]
  print 'direction=',u_dir
  start= pca.Mean
  while True:
    start2= np.array(start)-0.01*u_dir
    if not PointInPolygon2D(points, start2):  break
    start= start2
  print 'start=',start
  #direction= math.atan2(u_dir[1],u_dir[0])
  #rot= np.array([[math.cos(direction),-math.sin(direction)],[math.sin(direction),math.cos(direction)]])
  u_dir/= math.sqrt(u_dir[0]**2+u_dir[1]**2)
  rot= np.array([[u_dir[0],-u_dir[1]],[u_dir[1],u_dir[0]]])

  wave= TWaveGenerator()

  ##Test spreading wave (without planning)
  #fp= file('/tmp/spread1.dat','w')
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

  #Planning spreading wave (planning only)
  pstart= copy.deepcopy(start)
  while True:
    func= lambda ti,p1,p2: np.array(pstart) + np.dot(rot, np.array(wave.Evaluate(ti,m1=p1,m2=p2)))
    p1o,p2o= OptimizeWaveFunc1(func, p1_0=2.0, p2_0=2.0, eval_func=lambda f:EvalWaveFunc(f,points))
    #p1o,p2o= OptimizeWaveFunc2(func, p1_0=2.0, p2_0=2.0, eval_func=lambda f:EvalWaveFunc(f,points))
    if None in (p1o,p2o):  break
    wave.params.append([p1o, p2o])
    print p1o, p2o, EvalWaveFunc(lambda t:func(t,p1o,p2o),points), func(0.0,p1o,p2o), PointInPolygon2D(points,func(0.0,p1o,p2o))
    pstart= func(MAX_TIME,p1o,p2o)

  #Generate spreading trajectory
  fp= file('/tmp/spread2.dat','w')
  for t in FRange(0.0,MAX_TIME*float(len(wave.params)),500):
    p= np.array(start) + np.dot(rot, wave.Evaluate(t))
    fp.write(' '.join(map(str,p))+'\n')
  fp.close()

  print 'Plot by'
  print "qplot -x -s 'set size ratio -1' /tmp/orig.dat w l /tmp/spread2.dat w l"

