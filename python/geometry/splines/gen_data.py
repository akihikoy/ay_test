#!/usr/bin/python
import numpy as np
import numpy.linalg as la
import random
from math import sin,cos,tan,pi

#Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
#original_norm: keep original vec's norm, otherwise the norm is 1.
#Using The Gram-Schmidt process: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def Orthogonalize2(vec, base, original_norm=True):
  base= np.array(base)/la.norm(base)
  vec2= vec - np.dot(vec,base)*base
  if original_norm:  return vec2 / la.norm(vec2) * la.norm(vec)
  else:              return vec2 / la.norm(vec2)

#Get an orthogonal axis of a given axis
def GetOrthogonalAxis(axis):
  axis= np.array(axis)/la.norm(axis)
  if 1.0-abs(axis[2])>=1.0e-6:
    return Orthogonalize2([0.0,0.0,1.0],base=axis,original_norm=False)
  else:
    return [1.0,0.0,0.0]


#For 1-dimensional sample data: [t,x]*N

#Generate sample points: [t,x]*N
def Gen1d_1():
  return [[0,2],[1,3],[2.5,3.2],[3,1]]

#Generate sample points: [t,x]*N
def Gen1d_2(seed=10):
  random.seed(seed)
  data= [[random.random()*3.0,random.random()*2.0-1.0] for i in range(10)]
  data.sort()
  return data

#Generate sample points: [t,x]*N
def Gen1d_3():
  return [[i/10.0,sin(pi*i/10.0)] for i in range(21)]


#For 1-dimensional cyclic sample data: [t,x]*N
#where the first data and the last data are identical

#Generate cyclic sample points: [t,x]*N
def Gen1d_cyc1():
  return [[0,2],[1,3],[2.5,3.2],[3,1],[4,2]]

#Generate cyclic sample points: [t,x]*N
def Gen1d_cyc2(seed=10):
  data= Gen1d_2(seed)
  data[-1][1]= data[0][1]
  return data

#Generate cyclic sample points: [t,x]*N
def Gen1d_cyc3():
  return Gen1d_3()


#For 2-dimensional sample data: [t,x,y]*N

#Generate sample points: [t,x,y]*N
def Gen2d_1():
  return [[0,0,1],[1,1.5,2],[2.5,2.5,2.5],[3,3,2]]

#Generate sample points: [t,x,y]*N
def Gen2d_2():
  #random.seed(10)
  rand= lambda: random.random()*2.0-1.0
  data= [[random.random()*3.0,rand(),rand()] for i in range(10)]
  data.sort()
  return data

#Generate sample points: [t,x,y]*N
def Gen2d_3():
  vx= 0.1
  return [[0.0 ,vx*0.0 , 0.0],
          [0.25,vx*0.25, 1.0],
          [0.75,vx*0.75,-1.0],
          [1.0 ,vx*1.0 , 0.0]]


#For 2-dimensional cyclic sample data: [t,x,y]*N

#Generate cyclic sample points: [t,x,y]*N
def Gen2d_cyc1():
  return [[0,0,1],[1,1.5,2],[2.5,2.5,2.5],[3,3,2],[4,0,1]]

#Generate cyclic sample points: [t,x,y]*N
def Gen2d_cyc2():
  data= Gen2d_2()
  data[-1][1]= data[0][1]
  data[-1][2]= data[0][2]
  return data

#Generate sample points: [t,x,y]*N
def Gen2d_cyc3():
  vx= 0.1
  return [[0.0,-1.0, 0.0],
          [1.0, 0.0, 1.0],
          [2.0, 1.0, 0.0],
          [3.0, 0.0,-1.0],
          [4.0,-1.0, 0.0]]



#For 3-dimensional sample data: [t,x,y,z]*N

#Generate sample points: [t,x,y,z]*N
def Gen3d_1():
  return [[0,0,1,2],[1,1.5,2,3],[2.5,2.5,2.5,3.2],[3,3,2,1]]

#Generate sample points: [t,x,y,z]*N
def Gen3d_2():
  x0= np.array([0.0,1.0,2.0])
  xf= np.array([3.0,2.0,1.0])

  ex= xf-x0
  ex= np.array(ex)/la.norm(ex)
  ez= GetOrthogonalAxis(ex)
  ey= np.cross(ez,ex)

  #2 variable parameterization
  #dist= 1.0
  #angle= pi*0.0
  #xm= 0.5*(x0+xf)+dist*sin(angle)*ey+dist*cos(angle)*ez
  #data= []
  #data.append([0.0]+x0.tolist())
  #data.append([0.5]+xm.tolist())
  #data.append([1.0]+xf.tolist())

  #5 variable parameterization
  #angle= pi*0.0
  #param= [[0.1,1.5],[0.8,0.5]]
  #x= [[]]*2
  #for i in range(len(param)):
    #p= param[i]
    #x[i]= (1.0-p[0])*x0+p[0]*xf+p[1]*sin(angle)*ey+p[1]*cos(angle)*ez
  #data= []
  #data.append([0.0]+x0.tolist())
  #for i in range(len(param)):
    #data.append([param[i][0]]+x[i].tolist())
  #data.append([1.0]+xf.tolist())

  #5 variable parameterization(2)
  angle= pi*0.25
  param= [[1.0,pi*0.2],[0.5,pi*0.1]]
  #param= [[1.0,pi*0.0],[0.5,pi*0.0]]
  diff= lambda p1,p2: 0.5*la.norm(xf-x0) * (p1*ex*cos(p2) + p1*(sin(angle)*ey+cos(angle)*ez)*sin(p2))
  x= [[]]*2
  x[0]= x0 + diff(param[0][0],param[0][1])
  x[1]= xf + diff(param[1][0],pi-param[1][1])
  dt= np.array([la.norm(x0-x[0]),la.norm(x[0]-x[1]),la.norm(x[1]-xf)])
  dt= dt/sum(dt)
  data= []
  data.append([0.0]        +x0.tolist())
  data.append([dt[0]]      +x[0].tolist())
  data.append([dt[0]+dt[1]]+x[1].tolist())
  data.append([1.0]        +xf.tolist())
  return data

#Generate sample points: [t,x,y,z]*N
def Gen3d_3():
  #random.seed(10)
  rand= lambda: random.random()*2.0-1.0
  data= [[random.random()*3.0,rand(),rand(),rand()] for i in range(10)]
  data.sort()
  return data

#Generate sample points: [t,x,y,z]*N
def Gen3d_4():
  data= [[i/10.0,sin(pi*i/20.0),cos(pi*i/5.0),tan(pi*i/80.0)] for i in range(21)]
  return data

#Generate sample points: [t,x,y,z]*N
def Gen3d_5():
  return [[0.0,0.0,0.0,0.0],
          [0.5, 0.05,0.02,0.0],
          [1.5,-0.05,0.07,0.0],
          [2.5, 0.05,0.12,0.0],
          [3.5,-0.05,0.17,0.0],
          [4.0, 0.0,0.19,0.0]]


