#!/usr/bin/python3

import gpr_test2
import gpr_lin1
import lwr.lwr_test2 as lwr_test2

Mat= gpr_test2.Mat
FRange1= gpr_test2.FRange1
Rand= gpr_test2.Rand
np= gpr_test2.np

def AddOne(x):
  if x.shape[0]==1:  return Mat(x.tolist()[0]+[1.0])
  if x.shape[1]==1:  return Mat(x.T.tolist()[0]+[1.0]).T
  return None

def AddOnes(X):
  Y= X.tolist()
  for x in Y:
    x.append(1.0)
  return Mat(Y)

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math,time
  true_func= lambda x: 1.2+math.sin(x)
  data_x= [[x+1.0*Rand()] for x in FRange1(-3.0,5.0,3)]
  data_y= [[true_func(x[0])+0.3*Rand()] for x in data_x]

  fp1= open('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f\n' % (x[0],y[0]))
  fp1.close()

  fp1= open('/tmp/true.dat','w')
  for x in FRange1(-7.0,10.0,200):
    fp1.write('%f %f\n' % (x,true_func(x)))
  fp1.close()

  t1= time.time()
  lwr= lwr_test2.TLWR()
  lwr.Train(AddOnes(Mat(data_x)), data_y, c_min=0.3)
  print('lwr:learn time:',time.time()-t1)
  t1= time.time()
  gpr= gpr_test2.TGPR2()
  gpr.Train(data_x, data_y, c_min=0.5)
  print('gpr:learn time:',time.time()-t1)
  t1= time.time()
  gprlin= gpr_lin1.TGPRLin()
  gprlin.Train(data_x, data_y, c_min=0.5)
  print('gprlin:learn time:',time.time()-t1)

  models= [lwr, gpr, gprlin]
  names= ['lwr', 'gpr', 'gprlin']
  xmods= [lambda x:AddOne(x), lambda x:x, lambda x:x]

  for model,name,xmod in zip(models,names,xmods):
    fp2= open('/tmp/est_%s.dat'%name,'w')
    t1= time.time()
    for x in FRange1(-7.0,10.0,200):
      y= model.Predict(xmod(np.mat([x])))
      fp2.write('%f %f\n' % (x,y))
    print('%s:predict time:'%name,time.time()-t1)
    fp2.close()

  print('Plot by:')
  print('qplot -x /tmp/smpl.dat w p pt 5 /tmp/true.dat w l -cs \'w l\' /tmp/est_*.dat')
