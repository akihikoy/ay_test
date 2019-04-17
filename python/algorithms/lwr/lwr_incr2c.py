#!/usr/bin/python

from lwr_incr2 import *
#from lwr_incr3 import *

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

def Median(array):
  if len(array)==0:  return None
  a_sorted= copy.deepcopy(array)
  a_sorted.sort()
  return a_sorted[len(a_sorted)/2]

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  model= TLWR()
  #model.Init(c_min=0.6, f_reg=0.00001)
  #model.Init(c_min=0.3, f_reg=0.001)
  model.Init(c_min=0.01, f_reg=0.001)
  #model.Init(c_min=0.002, f_reg=0.001)
  #model.Init(c_min=0.0001, f_reg=0.0000001)
  src_file= 'data/ode_f2_smp.dat'; dim= [2,5,2]

  fp= file(src_file)
  while True:
    line= fp.readline()
    if not line: break
    data= line.split()
    model.Update(map(float,data[sum(dim[0:1]):sum(dim[0:2])]),
                 map(float,data[sum(dim[0:2]):sum(dim[0:3])]))
  #model.C= [0.01]*len(model.C)
  #model.C= model.AutoWidth(model.CMin)

  mi= [min([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]
  ma= [max([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]
  me= [Median([x[d] for x in model.DataX]) for d in range(len(model.DataX[0]))]

  #"""
  fp= open('/tmp/lwr/f2_est.dat','w')
  for x1 in FRange1(mi[2],ma[2],50):
    for x2 in FRange1(mi[4],ma[4],50):
      x= [me[0],me[1],x1,me[3],x2]
      pred= model.Predict(x,with_var=True)
      fp.write('%s\n' % ToStr(x,ToList(pred.Y),ToList(pred.Y.T+np.sqrt(np.diag(pred.Var)))))
    fp.write('\n')
  fp.close()
  fp= open('/tmp/lwr/f2_smp.dat','w')
  fp2= open('/tmp/lwr/f2_smpe.dat','w')
  for x,y in zip(model.DataX, model.DataY):
    fp.write('%s\n' % ToStr(x,y))
    pred= model.Predict(x,with_var=True)
    fp2.write('%s\n' % ToStr(x,ToList(pred.Y),ToList(pred.Y.T+np.sqrt(np.diag(pred.Var)))))
  fp.close()
  fp2.close()
  print '''qplot -x -3d -s 'set xlabel "flow_x";set ylabel "flow_var"' -cs 'u 3:5:6' /tmp/lwr/f2_est.dat w l /tmp/lwr/f2_smp.dat /tmp/lwr/f2_smpe.dat'''
  print '''qplot -x -3d -s 'set xlabel "flow_x";set ylabel "flow_var"' -cs 'u 3:5:7' /tmp/lwr/f2_est.dat w l /tmp/lwr/f2_smp.dat /tmp/lwr/f2_smpe.dat'''
  #"""

  """
  fp= open('/tmp/lwr/f2_est.dat','w')
  for x1 in FRange1(mi[1],ma[1],50):
    for x2 in FRange1(mi[3],ma[3],50):
      x= [me[0],x1,me[2],x2,me[4]]
      pred= model.Predict(x,with_var=True)
      fp.write('%s\n' % ToStr(x,ToList(pred.Y),ToList(pred.Y.T+np.sqrt(np.diag(pred.Var)))))
    fp.write('\n')
  fp.close()
  fp= open('/tmp/lwr/f2_smp.dat','w')
  fp2= open('/tmp/lwr/f2_smpe.dat','w')
  for x,y in zip(model.DataX, model.DataY):
    fp.write('%s\n' % ToStr(x,y))
    pred= model.Predict(x,with_var=True)
    fp2.write('%s\n' % ToStr(x,ToList(pred.Y),ToList(pred.Y.T+np.sqrt(np.diag(pred.Var)))))
  fp.close()
  fp2.close()
  print '''qplot -x -3d -s 'set xlabel "flow_x";set ylabel "flow_var"' -cs 'u 2:4:6' /tmp/lwr/f2_est.dat w l /tmp/lwr/f2_smp.dat /tmp/lwr/f2_smpe.dat'''
  print '''qplot -x -3d -s 'set xlabel "flow_x";set ylabel "flow_var"' -cs 'u 2:4:7' /tmp/lwr/f2_est.dat w l /tmp/lwr/f2_smp.dat /tmp/lwr/f2_smpe.dat'''
  #"""

