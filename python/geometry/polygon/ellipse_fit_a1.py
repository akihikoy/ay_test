#!/usr/bin/python3
#\file    ellipse_fit_a1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.09, 2020
import numpy as np
from ellipse_fit1 import EllipseFit2D    #SVD-based ellipse estimation
#from ellipse_fit2 import EllipseFit2D   #Fitzgibbon type direct ellipse fit (algebraic distance minimization with strict ellipse constraint, no regularization)
#from ellipse_fit3 import EllipseFit2D   #Regularized and scaled version of ef2 for improved numerical stability
#from ellipse_fit4 import EllipseFit2D   #Simple linear least squares fit without ellipse constraint
#from ellipse_fit5 import EllipseFit2D   #Bounded linear least squares enforcing semi-positive ellipse parameters
#from ellipse_fit6 import EllipseFit2D   #Ridge-regularized least squares fit with weak stability improvement
#from ellipse_fit7 import EllipseFit2D   #Nonlinear geometric-distance minimization using SVD-based initialization
#from ellipse_fit8 import EllipseFit2D   #SVD-based ellipse estimation using data covariance (center from mean)
#from ellipse_fit9 import EllipseFit2D   #SVD-based ellipse estimation using polygon centroid as center
from weighted_ellipse_fit2 import SampleWeightedEllipseFit2D


if __name__=='__main__':
  import gen_data
  XY= gen_data.Gen2d_02('rand')
  with open('/tmp/data.dat','w') as fp:
    for x,y in XY:
      fp.write('%f %f\n'%(x,y))

  #c,r1,r2,angle= EllipseFit2D(XY)
  c,r1,r2,angle= SampleWeightedEllipseFit2D(XY,[1.0]*len(XY))
  print('estimated:',c,r1,r2,angle)
  with open('/tmp/fit.dat','w') as fp:
    for th in np.linspace(0, 2*np.pi, 1000):
      x= c[0] + r1*np.cos(angle)*np.cos(th) - r2*np.sin(angle)*np.sin(th)
      y= c[1] + r1*np.sin(angle)*np.cos(th) + r2*np.cos(angle)*np.sin(th)
      fp.write('%f %f\n'%(x,y))

  print('#Plot by:')
  print('''qplot -x /tmp/data.dat /tmp/fit.dat w l''')

