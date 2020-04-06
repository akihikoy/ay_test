#!/usr/bin/python
#\file    2mlin2.py
#\brief   2-mode linear system (ver.2).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.13, 2020
import numpy as np

Sigmoid= lambda x,xt,beta: 1.0/(1.0+np.exp(-beta*(x-xt)))
LogSigmoid= lambda x,xt,beta: (lambda xx:np.log(1.0/(1.0+np.exp(xx))) if xx<0.0 else -xx+np.log(1.0/(1.0+np.exp(-xx))))(-beta*(x-xt))
#Two-mode linear model (parameter p=[xt,beta,f10,f11,f21])
F2MLin= lambda x,p: (p[3]-p[4])/p[1]*LogSigmoid(x,p[0],p[1])+p[4]*x+(p[2]-(p[4]-p[3])*p[0])

if __name__=='__main__':
  xt= 0.5
  beta= 50.0
  f10= 0.0
  f11= 1.0
  f21= -2.0
  p= [xt,beta,f10,f11,f21]
  with open('/tmp/p.dat','w') as fp:
    for x in np.arange(0.0,1.0,0.01):
      fp.write('{0} {1} {2} {3} {4}\n'.format(x, F2MLin(x,p), p[2]+p[3]*x, (p[2]+(p[3]-p[4])*p[0])+p[4]*x, Sigmoid(x,p[0],p[1])))
  print 'Plot by:'
  print 'qplot -x /tmp/p.dat w l'
  print 'qplot -x /tmp/p.dat u 1:5 w l /tmp/p.dat u 1:3 w l /tmp/p.dat u 1:4 w l /tmp/p.dat w l'
