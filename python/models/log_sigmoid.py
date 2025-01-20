#!/usr/bin/python3
#\file    log_sigmoid.py
#\brief   Stable computation for Log(Sigmoid(x)).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.14, 2020
import numpy as np

Sigmoid= lambda x,xt,beta: 1.0/(1.0+np.exp(-beta*(x-xt)))
LogSigmoid= lambda x,xt,beta: (lambda xx:np.log(1.0/(1.0+np.exp(xx))) if xx<0.0 else -xx+np.log(1.0/(1.0+np.exp(-xx))))(-beta*(x-xt))

if __name__=='__main__':
  xt= 0.5
  beta= 2000.0
  with open('/tmp/p.dat','w') as fp:
    for x in np.arange(0.0,1.0,0.01):
      fp.write('{0} {1} {2}\n'.format(x, np.log(Sigmoid(x,xt,beta)), LogSigmoid(x,xt,beta)))
  print('Plot by:')
  print('qplot -x /tmp/p.dat u 1:3 w l /tmp/p.dat u 1:2 w l')
