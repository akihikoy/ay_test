#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from ml.least_sq import Vec,GetWeightByLeastSq

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  #poss= []
  #vels= []
  flowpts= []
  fp= file('data/flowstat.dat')
  while True:
    line= fp.readline()
    if not line: break
    values= map(float,line.split())
    #poss.append(values[:3])
    #vels.append(values[3:])
    speed= la.norm(values[3:])
    if speed>0.02:
      flowpts.append(values)

  #flowpts: flow (moving) points

  quad_feat= lambda x: (1.0,x[0],x[0]**2)
  data_x= [[pt[2]] for pt in flowpts]
  data_f= [[pt[0], pt[1]] for pt in flowpts]
  W= GetWeightByLeastSq(data_x, data_f, quad_feat, f_reg=0.1)
  print W

  fp= file('/tmp/flowpts.dat','w')
  for pt in flowpts:
    fp.write('%s\n' % ' '.join(map(str,pt)))
  fp.close()

  fp= file('/tmp/flowfit.dat','w')
  zmin= min([pt[2] for pt in flowpts])
  zmax= max([pt[2] for pt in flowpts])
  for z in np.arange(zmin,zmax,(zmax-zmin)/50.0):
    xy= np.dot(W.T, quad_feat([z]))
    fp.write('%f %f %f\n' % (xy[0],xy[1],z))
  fp.close()

  print 'Plot by'
  print '''qplot -x -3d -s 'set ticslevel 0;set view equal xyz' /tmp/flowpts.dat pt 6 /tmp/flowpts.dat w vector /tmp/flowfit.dat w l'''

