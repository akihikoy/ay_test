#!/usr/bin/python
#\file    x_interpolation1.py
#\brief   Interpolation of two poses.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.05, 2018
from geom.geom import *
from geom.traj import QTrajToDQTraj, LimitQTrajVel

#Get a sequence of times, from 0 to dt including inum points (0 is not included).
def TimeTraj(dt, inum):
  return FRange1(0.0, dt, inum)[1:]

#Return the interpolation from x1 to x2 with N points
#p1 is not included
def XInterpolation(x1,x2,N):
  p1,R1= XToPosRot(x1)
  p2,R2= XToPosRot(x2)
  dp= (p2-p1)/float(N)
  trans_R= np.dot(R2,R1.T)
  w= InvRodrigues(trans_R)
  traj=[]
  for t in range(N):
    R= np.dot(Rodrigues(float(t+1)/float(N)*w),R1)
    p1= p1+dp
    traj.append(PosRotToX(p1,R))
  return traj


#Pseudo implementation of following a trajectory: output to a file.
def FollowXTraj(file_name, x_curr, x_traj, t_traj):
  LimitQTrajVel(x_curr, x_traj, t_traj, qvel_limits=[0.1]*7, acc_phase=9)

  #x_traj.insert(0, x_curr)
  #t_traj.insert(0, 0.0)
  dx_traj= QTrajToDQTraj(x_traj, t_traj)

  with open(file_name, 'w') as fd:
    for x,dx,t in zip(x_traj,dx_traj,t_traj):
      fd.write('{t} {x} {dx}\n'.format(t=t, x=' '.join(map(str,x)), dx=' '.join(map(str,dx)) ) )

#Pseudo implementation of moving to a target pose x_trg from x_curr.
def MoveToXI(file_name, x_curr, x_trg, dt=4.0, inum=30):
  x_traj= XInterpolation(x_curr,x_trg,inum)
  t_traj= TimeTraj(dt,inum)
  x_traj.insert(0, x_curr)
  t_traj.insert(0, 0.0)
  FollowXTraj(file_name, x_curr, x_traj, t_traj)


def Main():
  x1= [0.,0.,0., 0.,0.,0.,1.]
  x2= [1.,1.5,-1., 0.,0.,0.38268343,0.92387953]
  MoveToXI('/tmp/traj1.dat',x1,x2)

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands=[
    '''qplot -x2 aaa {f1} u 2:3 w lp &''',
    '''qplot -x2 aaa -s 'set xlabel "time"; set ylabel "position"'
        {f1} u 1:2 w lp t '"x"'
        {f1} u 1:3 w lp t '"y"'
        &''',
    '''qplot -x2 aaa -s 'set xlabel "time"; set ylabel "velocity"'
        {f1} u 1:9  w lp t '"dx"'
        {f1} u 1:10 w lp t '"dy"'
        &''',
    '''''',
    ]
  commands= [cmd.format(f1='/tmp/traj1.dat') for cmd in commands]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
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
