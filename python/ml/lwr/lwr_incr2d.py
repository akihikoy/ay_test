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

import math
#TrueFunc= lambda x: 1.2+math.sin(2.0*(x[0]+x[1]))
TrueFunc= lambda x: 0.1*(x[0]*x[0]+x[1]*x[1])
#TrueFunc= lambda x: 4.0-x[0]*x[1]
#TrueFunc= lambda x: 4.0-x[0]-x[1] if x[0]**2+x[1]**2<2.0 else 0.0

def GenData(n=100, noise=0.3):
  #data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,n)]
  data_x= [[Rand(-3.0,3.0), Rand(-3.0,3.0)] for k in range(n)]
  data_y= [[TrueFunc(x)+noise*Rand()] for x in data_x]
  return data_x, data_y


def Main():
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  data_x, data_y = GenData(20, noise=0.0)  #TEST: n samples, noise

  model= TLWR()
  #model.Init(c_min=0.6, f_reg=0.00001)
  #model.Init(c_min=0.3, f_reg=0.001)
  model.Init(c_min=0.01, f_reg=0.001)
  #model.Init(c_min=0.002, f_reg=0.001)
  #model.Init(c_min=0.0001, f_reg=0.0000001)

  for x,y in zip(data_x, data_y):
    model.Update(x,y)
  #model.C= [0.01]*len(model.C)
  #model.C= model.AutoWidth(model.CMin)

  nt= 25
  N_test= nt*nt
  x_test= np.array(sum([[[x1,x2] for x2 in FRange1(-3.0,3.0,nt)] for x1 in FRange1(-3.0,3.0,nt)],[])).astype(np.float32)
  y_test= np.array([[TrueFunc(x)] for x in x_test]).astype(np.float32)

  # Dump data for plot:
  fp1= file('/tmp/smpl_train2.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()
  # Dump data for plot:
  fp1= file('/tmp/smpl_test2.dat','w')
  for x,y,i in zip(x_test,y_test,range(len(y_test))):
    if i%(nt+1)==0:  fp1.write('\n')
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()

  pred= [[model.Predict(x).Y[0,0]] for x in x_test]
  fp1= file('/tmp/lwr_est.dat','w')
  for x,y,i in zip(x_test,pred,range(len(pred))):
    if i%(nt+1)==0:  fp1.write('\n')
    fp1.write('%s #%i# %s\n' % (' '.join(map(str,x)),len(x)+1,' '.join(map(str,y))))
  fp1.close()



def PlotGraphs():
  print 'Plotting graphs..'
  import os,sys
  opt= sys.argv[2:]
  commands=[
    '''qplot -x2 aaa -3d {opt} -s 'set xlabel "x";set ylabel "y";set ticslevel 0;'
          -cs 'u 1:2:4' /tmp/smpl_train2.dat pt 6 ps 2 t '"sample"'
          /tmp/smpl_test2.dat w l lw 3 t '"true"'
          /tmp/lwr_est.dat w l t '"LWR"' &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.format(opt=' '.join(opt)).splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
