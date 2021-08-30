#!/usr/bin/python

'''
ref.
http://vilkeliskis.com/blog/2013/09/08/machine_learning_part_2_locally_weighted_linear_regression.html
'''

from lwr_sample1 import *

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  import math
  true_func= lambda x: 1.2+math.sin(2.0*x[0])*x[1]
  data_x= [[4.0*Rand(),4.0*Rand(),1.0] for i in range(20)]  # ,1.0 is to learn const
  data_y= [[true_func(x[:2])+0.3*Rand()] for x in data_x]

  fp1= file('/tmp/smpl.dat','w')
  for x,y in zip(data_x,data_y):
    fp1.write('%f %f %f\n' % (x[0],x[1],y[0]))
  fp1.close()

  fp1= file('/tmp/true.dat','w')
  fp2= file('/tmp/est.dat','w')
  for x1 in FRange1(-4.0,4.0,50):
    for x2 in FRange1(-4.0,4.0,50):
      y= lwr_predict(data_x, data_y, np.mat([x1,x2,1.0]), c=0.5)  # ,1.0 is to learn const
      fp1.write('%f %f %f\n' % (x1,x2,true_func([x1,x2])))
      fp2.write('%f %f %f\n' % (x1,x2,y))
    fp1.write('\n')
    fp2.write('\n')
  fp1.close()
  fp2.close()

  print 'Plot by:'
  print 'qplot -x -3d /tmp/est.dat w l /tmp/true.dat w l /tmp/smpl.dat w p'

