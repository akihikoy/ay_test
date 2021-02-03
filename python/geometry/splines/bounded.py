#!/usr/bin/python
#\file    bounded.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.03, 2021
from cubic_hermite_spline import TCubicHermiteSpline
import numpy as np

if __name__=='__main__':
  #x_traj,t_traj= [0.07, 0.07, 0.008], [0.0, 0.2, 0.25]
  #x_traj,t_traj= [0.0, 0.0, 0.08, 0.15], [0.0, 0.2, 0.25, 0.30]
  x_traj,t_traj= [0.0, 0.0, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08], [0.0, 0.2, 0.25, 0.35, 0.40, 0.50, 0.55, 0.70]
  data= [[t,x] for t,x in zip(t_traj,x_traj)]
  spline= TCubicHermiteSpline()
  spline.Initialize(data, tan_method=spline.CARDINAL, c=1.0, m=0.0)
  #NOTE: Setting c to 1.0, spline does not overshoot.

  with open('/tmp/data.dat','w') as fp:
    for t,x in data:
      fp.write('{t} {x}\n'.format(t=t,x=x))
  with open('/tmp/spline.dat','w') as fp:
    for t in np.linspace(min(t_traj),max(t_traj),200):
      fp.write('{t} {x}\n'.format(t=t,x=spline.Evaluate(t,with_tan=False)))

  print '#Plot by:'
  print '''qplot -x /tmp/spline.dat w l /tmp/data.dat w p'''
