#!/usr/bin/python
#Cubic Hermite Spline

# Matlab-like mod function that returns always positive
import math
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)

#Generate a cubic Hermite spline from a key points.
#Key points: [[t0,x0],[t1,x1],[t2,x2],...].
class TCubicHermiteSpline:
  class TKeyPoint:
    T= 0.0  #Input
    X= 0.0  #Output
    M= 0.0  #Gradient
    def __str__(self):
      return '['+str(self.T)+', '+str(self.X)+', '+str(self.M)+']'

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
  #with_dd: If True, x,dx/dt,ddx/ddt are returned (with_tan is ignored).
  def Evaluate(self, t, with_tan=False, with_dd=False):
    idx= self.FindIdx(t,self.idx_prev)
    if abs(t-self.KeyPts[-1].T)<1.0e-6:  idx= len(self.KeyPts)-2
    if idx<0 or idx>=len(self.KeyPts)-1:
      print 'WARNING: Given t= %f is out of the key points (index: %i)' % (t,idx)
      if idx<0:
        idx= 0
        t= self.KeyPts[0].T
      else:
        idx= len(self.KeyPts)-2
        t= self.KeyPts[-1].T

    h00= lambda t: t*t*(2.0*t-3.0)+1.0
    h10= lambda t: t*(t*(t-2.0)+1.0)
    h01= lambda t: t*t*(-2.0*t+3.0)
    h11= lambda t: t*t*(t-1.0)

    self.idx_prev= idx
    p0= self.KeyPts[idx]
    p1= self.KeyPts[idx+1]
    dT= p1.T-p0.T
    tr= (t-p0.T) / dT
    x= h00(tr)*p0.X + h10(tr)*dT*p0.M + h01(tr)*p1.X + h11(tr)*dT*p1.M
    if not with_tan and not with_dd:  return x

    dh00= lambda t: t*(6.0*t-6.0)
    dh10= lambda t: t*(3.0*t-4.0)+1.0
    dh01= lambda t: t*(-6.0*t+6.0)
    dh11= lambda t: t*(3.0*t-2.0)
    dx= (dh00(tr)*p0.X + dh10(tr)*dT*p0.M + dh01(tr)*p1.X + dh11(tr)*dT*p1.M) / dT
    if not with_dd:  return x,dx

    ddh00= lambda t: 12*t - 6
    ddh10= lambda t: 6*t - 4
    ddh01= lambda t: -12*t + 6
    ddh11= lambda t: 6*t - 2
    ddx= (ddh00(tr)*p0.X + ddh10(tr)*dT*p0.M + ddh01(tr)*p1.X + ddh11(tr)*dT*p1.M) / (dT*dT)
    return x,dx,ddx

  #Compute a phase information (n, tp) for a cyclic spline curve.
  #n:  n-th occurrence of the base wave
  #tp: phase (time in the base wave)
  def PhaseInfo(self, t):
    t0= self.KeyPts[0].T
    te= self.KeyPts[-1].T
    T= te-t0
    mod= Mod(t-t0,T)
    tp= t0+mod  #Phase
    n= (t-t0-mod)/T
    return n, tp

  #Return interpolated value at t (cyclic version).
  #pi: Phase information.
  #with_tan: If True, both x and dx/dt are returned.
  #with_dd: If True, x,dx/dt,ddx/ddt are returned (with_tan is ignored).
  def EvaluateC(self, t, pi=None, with_tan=False, with_dd=False):
    if pi is None:
      n, tp= self.PhaseInfo(t)
    else:
      n, tp= pi
    if with_dd:  x,dx,ddx= self.Evaluate(tp, with_dd=with_dd)
    else:
      if with_tan:  x,dx= self.Evaluate(tp, with_tan=with_tan)
      else:        x= self.Evaluate(tp)
    x= x + n*(self.KeyPts[-1].X - self.KeyPts[0].X)
    return (x,dx,ddx) if with_dd else ( (x,dx) if with_tan else x )

  #data= [[t0,x0],[t1,x1],[t2,x2],...]
  FINITE_DIFF=0  #Tangent method: finite difference method
  CARDINAL=1  #Tangent method: Cardinal spline (c is used)
  ZERO= 0  #End tangent: zero
  GRAD= 1  #End tangent: gradient (m is used)
  CYCLIC= 2  #End tangent: treating data as cyclic (KeyPts[-1] and KeyPts[0] are considered as an identical point)
  def Initialize(self, data, tan_method=CARDINAL, end_tan=GRAD, c=0.0, m=1.0):
    if data != None:
      self.KeyPts= [self.TKeyPoint() for i in range(len(data))]
      for idx in range(len(data)):
        self.KeyPts[idx].T= data[idx][0]
        self.KeyPts[idx].X= data[idx][1]

    #Store parameters for future use / remind parameters if not given
    if tan_method is None:  tan_method= self.Param.TanMethod
    else:                   self.Param.TanMethod= tan_method
    if end_tan is None:  end_tan= self.Param.EndTan
    else:                self.Param.EndTan= end_tan
    if c is None:  c= self.Param.C
    else:          self.Param.C= c
    if m is None:  m= self.Param.M
    else:          self.Param.M= m

    grad= lambda idx1,idx2: (self.KeyPts[idx2].X-self.KeyPts[idx1].X)/(self.KeyPts[idx2].T-self.KeyPts[idx1].T)

    if tan_method == self.FINITE_DIFF:
      for idx in range(1,len(self.KeyPts)-1):
        self.KeyPts[idx].M= 0.5*grad(idx,idx+1) + 0.5*grad(idx-1,idx)
    elif tan_method == self.CARDINAL:
      for idx in range(1,len(self.KeyPts)-1):
        self.KeyPts[idx].M= (1.0-c)*grad(idx-1,idx+1)

    if end_tan == self.ZERO:
      self.KeyPts[0].M= 0.0
      self.KeyPts[-1].M= 0.0
    elif end_tan == self.GRAD:
      self.KeyPts[0].M= m*grad(0,1)
      self.KeyPts[-1].M= m*grad(-2,-1)
    elif end_tan == self.CYCLIC:
      if tan_method == self.FINITE_DIFF:
        grad_p1= grad(0,1)
        grad_n1= grad(-2,-1)
        M= 0.5*grad_p1 + 0.5*grad_n1
        self.KeyPts[0].M= M
        self.KeyPts[-1].M= M
      elif tan_method == self.CARDINAL:
        T= self.KeyPts[-1].T - self.KeyPts[0].T
        X= self.KeyPts[-1].X - self.KeyPts[0].X
        grad_2= (X+self.KeyPts[1].X-self.KeyPts[-2].X)/(T+self.KeyPts[1].T-self.KeyPts[-2].T)
        M= (1.0-c)*grad_2
        self.KeyPts[0].M= M
        self.KeyPts[-1].M= M

  def Update(self):
    self.Initialize(data=None, tan_method=None, end_tan=None, c=None, m=None)


if __name__=="__main__":
  import gen_data
  spline= TCubicHermiteSpline()

  data= gen_data.Gen1d_1()
  #data= gen_data.Gen1d_2()
  #data= gen_data.Gen1d_3()

  spline.Initialize(data, tan_method=spline.CARDINAL, c=0.0)
  #spline.KeyPts[0].M= 5.0
  #spline.KeyPts[-1].M= -5.0
  pf= file('/tmp/spline1.dat','w')
  t= data[0][0]
  while t<data[-1][0]:
    pf.write('%f %s\n' % (t, ' '.join(map(str,spline.Evaluate(t,with_dd=True)))))
    t+= 0.001
  print 'Generated:','/tmp/spline1.dat'

  pf= file('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%f %f\n' % (d[0],d[1]))
  print 'Generated:','/tmp/spline0.dat'

  print 'Plot by:'
  print 'qplot -x -s "set ytics nomirror; set y2tics" /tmp/spline1.dat w l t "\'x\'" /tmp/spline0.dat w p pt 5 ps 2 t "\'given points\'"'
  print 'qplot -x -s "set ytics nomirror; set y2tics" /tmp/spline1.dat w l t "\'x\'" /tmp/spline1.dat u 1:3 ax x1y2 w l t "\'dx\'" /tmp/spline0.dat w p pt 5 ps 2 t "\'given points\'"'
  print 'qplot -x -s "set ytics nomirror; set y2tics" /tmp/spline1.dat w l t "\'x\'" /tmp/spline1.dat u 1:3 ax x1y2 w l t "\'dx\'" /tmp/spline1.dat u 1:4 ax x1y2 w l t "\'ddx\'" /tmp/spline0.dat w p pt 5 ps 2 t "\'given points\'"'
