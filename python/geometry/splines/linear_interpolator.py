#!/usr/bin/python3
#\file    linear_interpolator.py
#\brief   Linear interpolator class with the same interface as TCubicHermiteSpline.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.18, 2025
from cubic_hermite_spline import Mod

#Generate a linear interpolator from a list of key points.
#Key points: [[t0,x0],[t1,x1],[t2,x2],...].
#This class has the same interface as TCubicHermiteSpline.
class TLinearInterpolator:
  class TKeyPoint:
    T= 0.0  #Input
    X= 0.0  #Output
    def __str__(self):
      return '['+str(self.T)+', '+str(self.X)+']'

  class TParam: pass

  def __init__(self):
    self.idx_prev= 0
    self.Param= self.TParam()

  def FindIdx(self, t, idx_prev=0):
    idx= idx_prev
    if idx>=len(self.KeyPts): idx= len(self.KeyPts)-1
    while idx+1<len(self.KeyPts) and t>self.KeyPts[idx+1].T:  idx+=1
    while idx>=0 and t<self.KeyPts[idx].T:  idx-=1
    return idx

  #Return interpolated value at t.
  #with_tan: If True, both x and dx/dt are returned.
  #with_dd: If True, x,dx/dt,ddx/ddt are returned (here ddx=0).
  def Evaluate(self, t, with_tan=False, with_dd=False):
    idx= self.FindIdx(t,self.idx_prev)
    if abs(t-self.KeyPts[-1].T)<1.0e-6:  idx= len(self.KeyPts)-2
    if idx<0 or idx>=len(self.KeyPts)-1:
      print('WARNING: Given t= %f is out of the key points (index: %i)' % (t,idx))
      if idx<0:
        idx= 0
        t= self.KeyPts[0].T
      else:
        idx= len(self.KeyPts)-2
        t= self.KeyPts[-1].T

    self.idx_prev= idx
    p0= self.KeyPts[idx]
    p1= self.KeyPts[idx+1]
    dT= p1.T-p0.T
    tr= (t-p0.T) / dT
    x= (1.0-tr)*p0.X + tr*p1.X
    if not with_tan and not with_dd:  return x

    dx= (p1.X-p0.X)/dT
    if not with_dd:  return x,dx

    ddx= 0.0  #Second derivative is always zero in linear interpolation
    return x,dx,ddx

  #Compute a phase information (n, tp) for a cyclic curve.
  def PhaseInfo(self, t):
    t0= self.KeyPts[0].T
    te= self.KeyPts[-1].T
    T= te-t0
    mod= Mod(t-t0,T)
    tp= t0+mod  #Phase
    n= (t-t0-mod)/T
    return n, tp

  #Return interpolated value at t (cyclic version).
  def EvaluateC(self, t, pi=None, with_tan=False, with_dd=False):
    if pi is None:
      n, tp= self.PhaseInfo(t)
    else:
      n, tp= pi
    if with_dd:  x,dx,ddx= self.Evaluate(tp, with_dd=with_dd)
    else:
      if with_tan:  x,dx= self.Evaluate(tp, with_tan=with_tan)
      else:         x= self.Evaluate(tp)
    x= x + n*(self.KeyPts[-1].X - self.KeyPts[0].X)
    return (x,dx,ddx) if with_dd else ( (x,dx) if with_tan else x )

  #data= [[t0,x0],[t1,x1],[t2,x2],...]
  def Initialize(self, data):
    if data != None:
      self.KeyPts= [self.TKeyPoint() for i in range(len(data))]
      for idx in range(len(data)):
        self.KeyPts[idx].T= data[idx][0]
        self.KeyPts[idx].X= data[idx][1]

  def Update(self):
    self.Initialize(data=None)


