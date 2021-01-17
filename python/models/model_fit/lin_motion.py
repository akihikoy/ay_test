#!/usr/bin/python
#\file    lin_motion.py
#\brief   Identifying linear motion from position observations.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.18, 2021
import numpy as np

#Estimate the current position and velocity from time-position sequence
#where we assume the linear constant velocity.
#  tp_seq: List of [time,position] where position is 3d vector.
def GetPosVelFromTimePosSeq(tp_seq):
  if len(tp_seq)<2:  return None
  v_seq= [(tp1[1]-tp0[1])/(tp1[0]-tp0[0]) for tp1,tp0 in zip(tp_seq[1:],tp_seq[:-1])]
  vel= np.mean(v_seq,axis=0)
  t_curr= tp_seq[-1][0]
  pos_seq= [p+(t_curr-t)*vel for t,p in tp_seq]
  pos= np.mean(pos_seq,axis=0)
  return pos,vel

if __name__=='__main__':
  vel= np.random.uniform([-2]*3,[2]*3)
  pos0= np.array([2.,3.,1.])
  t0= 10.
  tp_seq= []
  for t_curr in np.linspace(t0,t0+3.,10):
    pos_curr= pos0+(t_curr-t0)*vel
    tp_seq.append([t_curr,pos_curr+np.random.uniform([-0.1]*3,[0.1]*3)])

  pos_est,vel_est= GetPosVelFromTimePosSeq(tp_seq)
  print 'vel=',vel
  print 'pos_curr=',pos_curr
  print 'vel_est=',vel_est,np.linalg.norm(vel-vel_est)
  print 'pos_est=',pos_est,np.linalg.norm(pos_curr-pos_est)
  with open('/tmp/motion_data.dat','w') as fp:
    for t_curr,p_data in tp_seq:
      fp.write('{0} {1}\n'.format(t_curr,' '.join(map(str,p_data))))
  with open('/tmp/motion_true.dat','w') as fp:
    for t_curr,p_data in tp_seq:
      p_true= pos0+(t_curr-t0)*vel
      fp.write('{0} {1}\n'.format(t_curr,' '.join(map(str,p_true))))
  with open('/tmp/motion_est.dat','w') as fp:
    for t_curr,p_data in tp_seq:
      p_est= pos_est+(t_curr-tp_seq[-1][0])*vel_est
      fp.write('{0} {1}\n'.format(t_curr,' '.join(map(str,p_est))))
  print '''Plot by:'''
  print '''qplot -3d -x /tmp/motion_true.dat u 2:3:4 w l /tmp/motion_data.dat u 2:3:4 w p /tmp/motion_est.dat u 2:3:4 w lp'''
