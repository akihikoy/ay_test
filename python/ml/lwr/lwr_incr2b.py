#!/usr/bin/python

from lwr_incr2 import *

#From gradient/toy1.py
class TContainer: pass
sys= TContainer()
sys.yl= 0.5  #Location of receiving container.
sys.wl= 0.3  #Size of receiving container
sys.DEBUG= False
sys.bound= [[0.2,0.3],[0.8,0.8]]  #Boundary of y,z
sys.bound2= [[0.2,0.0,0.0],[0.8,0.0,0.8]]  #Boundary of cy,cz,w
sys.n_dyn= 0  #Number of Dynamics computation
def Vec(x):
  return np.array(x)
def Assess(sys,cy,cz,w):
  if (cy+0.5*w) >= (sys.yl+0.5*sys.wl):
    return (sys.yl+0.5*sys.wl) - (cy+0.5*w)  #Penalty
    #return -1
  if (cy-0.5*w) <= (sys.yl-0.5*sys.wl):
    return (cy-0.5*w) - (sys.yl-0.5*sys.wl)  #Penalty
    #return -1
  e= +1.0 - 50.0*(cy-sys.yl)**2
  return e if e>0.0 else e
  #return +1.0
def F2(x2):
  return Vec([Assess(sys,x2[0],x2[1],x2[2])]).T
def ToStr(*lists):
  s= ''
  delim= ''
  for v in lists:
    s+= delim+' '.join(map(str,list(v)))
    delim= ' '
  return s
def ToList(x):
  if x==None:  return []
  elif isinstance(x,list):  return x
  elif isinstance(x,(np.ndarray,np.matrix)):
    if len(x.shape)==1:  return x.tolist()
    if len(x.shape)==2:
      if x.shape[0]==1:  return x.tolist()[0]
      if x.shape[1]==1:  return x.T.tolist()[0]
  print 'ToList: x=',x
  raise Exception('ToList: Impossible to serialize:',x)

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  model= TLWR()
  #model.Init(c_min=0.3, f_reg=0.001)
  model.Init(c_min=0.01, f_reg=0.001)
  #model.Init(c_min=0.002, f_reg=0.001)
  #model.Init(c_min=0.001, f_reg=0.001)
  #src_file= 'data/f2_smp.dat'
  #src_file= 'data/f2_smp4.dat'
  src_file= 'data/f2_smp5.dat'

  #"""
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data= map(float,line.split())
    model.Update(data[:3],[data[3]])
  #model.C= [0.01]*len(model.C)
  #model.C= model.AutoWidth(model.CMin)

  fp= file('/tmp/f2.dat','w')
  for cy in FRange1(sys.bound2[0][0],sys.bound2[1][0],50):
    for w in FRange1(sys.bound2[0][2],sys.bound2[1][2],50):
      x2= [cy,0.5*(sys.bound2[0][1]+sys.bound2[1][1]),w]
      fp.write('%s\n' % ToStr(x2,F2(x2)))
    fp.write('\n')
  fp.close()
  fp= file('/tmp/f2_est.dat','w')
  for cy in FRange1(sys.bound2[0][0],sys.bound2[1][0],50):
    for w in FRange1(sys.bound2[0][2],sys.bound2[1][2],50):
      x2= [cy,0.5*(sys.bound2[0][1]+sys.bound2[1][1]),w]
      pred= model.Predict(x2,with_var=True)
      fp.write('%s\n' % ToStr(x2,ToList(pred.Y),ToList(pred.Y+pred.Var)))
    fp.write('\n')
  fp.close()
  fp= file('/tmp/f2_smp.dat','w')
  for x2,y in zip(model.DataX, model.DataY):
    fp.write('%s\n' % ToStr(x2,y))
  fp.close()
  print '''qplot -x -3d -cs 'u 1:3:4' /tmp/f2.dat w l /tmp/f2_est.dat w l /tmp/f2_smp.dat'''
  print '''qplot -x -3d -cs 'u 1:3:4' /tmp/f2.dat w l /tmp/f2_est.dat w l /tmp/f2_smp.dat -cs '' /tmp/f2_est.dat u 1:3:5 w l'''
  #"""

  """
  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data= map(float,line.split())
    model.Update([data[0]],[data[3]])
  #model.C= [0.1]*len(model.C)

  fp= file('/tmp/f2.dat','w')
  for cy in FRange1(sys.bound2[0][0],sys.bound2[1][0],200):
    #for w in FRange1(sys.bound2[0][2],sys.bound2[1][2],50):
    x2= [cy]
    fp.write('%s\n' % ToStr(x2,F2([cy,0.0,0.0])))
    #fp.write('\n')
  fp.close()
  fp= file('/tmp/f2_est.dat','w')
  for cy in FRange1(sys.bound2[0][0],sys.bound2[1][0],200):
    #for w in FRange1(sys.bound2[0][2],sys.bound2[1][2],50):
    x2= [cy]
    pred= model.Predict(x2,with_var=True)
    fp.write('%s\n' % ToStr(x2,ToList(pred.Y),ToList(pred.Var)))
    #fp.write('\n')
  fp.close()
  fp= file('/tmp/f2_smp.dat','w')
  for x2,y in zip(model.DataX, model.DataY):
    fp.write('%s\n' % ToStr(x2,y))
  fp.close()
  print '''qplot -x -cs 'u 1:2' /tmp/f2.dat w l /tmp/f2_est.dat w l /tmp/f2_smp.dat'''
  #"""

