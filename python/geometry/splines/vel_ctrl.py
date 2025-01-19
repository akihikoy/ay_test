#!/usr/bin/python3
import numpy as np
import numpy.linalg as la

def Sign(x):
  if x==0.0:   return 0
  elif x>0.0:  return +1
  elif x<0.0:  return -1

#Modify the velocity of a given trajectory (base routine).
#t0: Current internal time maintaining a playing point of the trajectory
#v: Target speed
#traj: Function to map time t --> point x
#time_step: Control time step (actual value)
#T: Maximum internal time
#num_iter_l: Limit of iteration number (linear search)
#num_iter_e: Limit of iteration number (exponential search)
#norm: Function to compute norm of two vectors
#diff: Function to compute difference of two vectors
#is_samedir: Function to judge if two vectors have the same direction
def modify_traj_velocity_base(t0, v, traj, time_step, T, num_iter_l, num_iter_e, norm, diff, is_samedir):
  num_iter_l= int(num_iter_l)
  num_iter_e= int(num_iter_e)
  ddt= time_step/(float(num_iter_l)/5.0)
  v_dt= v*time_step
  x0= traj(t0)
  t1= t0
  x1= x0
  s1= None
  cross= False
  #print '-----------'
  while num_iter_l>0 and num_iter_e>0 and t1<T:
    t2= t1+ddt
    if t2>T:  t2= T
    x2= traj(t2)
    dx= norm(x2,x0)
    s2= diff(x2,x1)
    if s1==None:  s1= s2
    #print t2,dx/time_step,s1,s2
    samedir= is_samedir(s1,s2)
    if dx<v_dt and samedir:
      t1= t2
      x1= x2
      num_iter_l-= 1
    else:
      if dx<v_dt and not samedir:
        #print 'cross#',t2,x2
        cross= True
        over_t2= t2
        over_x2= x2
      ddt*= 0.5
      num_iter_e-= 1
  if cross:
    #print 'cross',over_t2,over_x2,num_iter_l,num_iter_e
    t1= over_t2
    x1= over_x2
  #print t1,t1-t0, x1, norm(x1,x0)/time_step, v,num_iter_l, num_iter_e
  return t1, x1, norm(x1,x0)/time_step

#Modify the velocity of a given trajectory (1-d version).
#t0: Current internal time maintaining a playing point of the trajectory
#v: Target speed
#traj: Function to map time t --> point x
#time_step: Control time step (actual value)
#T: Maximum internal time
#num_iter_l: Limit of iteration number (linear search)
#num_iter_e: Limit of iteration number (exponential search)
def ModifyTrajVelocity(t0, v, traj, time_step, T, num_iter_l=50, num_iter_e=6):
  norm= lambda x2,x1: abs(x2-x1)
  diff= lambda x2,x1: x2-x1
  is_samedir= lambda s1,s2: Sign(s1)==Sign(s2)
  return modify_traj_velocity_base(t0, v, traj, time_step, T, num_iter_l, num_iter_e, norm, diff, is_samedir)

#Modify the velocity of a given trajectory (N-d version).
#t0: Current internal time maintaining a playing point of the trajectory
#v: Target speed
#trajs: List of functions to map time t --> point x
#time_step: Control time step (actual value)
#T: Maximum internal time
#num_iter_l: Limit of iteration number (linear search)
#num_iter_e: Limit of iteration number (exponential search)
def ModifyTrajVelocityV(t0, v, traj, time_step, T, num_iter_l=50, num_iter_e=6):
  diff= lambda x2,x1: [x2d-x1d for (x2d,x1d) in zip(x2,x1)]
  norm= lambda x2,x1: la.norm(diff(x2,x1))
  is_samedir= lambda s1,s2: np.dot(s1,s2) > 0
  return modify_traj_velocity_base(t0, v, traj, time_step, T, num_iter_l, num_iter_e, norm, diff, is_samedir)


if __name__=='__main__':
  def PrintEq(s):  print('%s= %r' % (s, eval(s)))
  from cubic_hermite_spline import TCubicHermiteSpline
  import gen_data
  import math
  spline= TCubicHermiteSpline()

  data= gen_data.Gen1d_1()
  #data= gen_data.Gen1d_2()
  #data= gen_data.Gen1d_3()

  spline.Initialize(data, tan_method=spline.CARDINAL, c=0.0)
  dt= 0.005

  pf= open('/tmp/spline1.dat','w')
  t= data[0][0]
  while t<data[-1][0]:
    pf.write('%f %f\n' % (t, spline.Evaluate(t)))
    t+= dt
  print('Generated:','/tmp/spline1.dat')

  pf= open('/tmp/spline2.dat','w')
  t= data[0][0]
  ti= data[0][0]  #Internal time
  T= data[-1][0]  #Length
  v_trg= 1.0
  while ti<T:
    #v_trg= (1.0+math.cos(t))
    v_trg= 0.1*(1.0+2.0*math.cos(3.0*t)); v_trg= 0.0 if v_trg<0.0 else v_trg
    ti,x,v= ModifyTrajVelocity(ti, v_trg, lambda s:spline.Evaluate(s), dt, T)
    pf.write('%f %f %f %f\n' % (t, x, v, v_trg))
    t+= dt
  print('Generated:','/tmp/spline2.dat')

  pf= open('/tmp/spline0.dat','w')
  for d in data:
    pf.write('%f %f\n' % (d[0],d[1]))
  print('Generated:','/tmp/spline0.dat')


  print('Plot by:')
  print('qplot -x /tmp/spline1.dat w l /tmp/spline0.dat w p pt 5 ps 2 /tmp/spline2.dat w l /tmp/spline2.dat u 1:4 w l /tmp/spline2.dat u 1:3 w lp')
