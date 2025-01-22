#!/usr/bin/python3
#\file    grip2.py
#\brief   We investigate the relation between command (0-255) and position (meters).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.07, 2015

import roslib
import rospy
from robotiq import TRobotiq

if __name__=='__main__':
  rospy.init_node('robotiq_test')
  rq= TRobotiq()
  rq.Init()

  while not rospy.is_shutdown():
    c= input('Type 0-255 or {o,c,q} > ')
    if c=='q':  break
    elif c=='o':  rq.OpenGripper(blocking=True)
    elif c=='c':  rq.CloseGripper(blocking=True)
    else:
      try:
        cmd= int(c)
        rq.MoveGripper(pos=cmd, max_effort=0, speed=0, blocking=True)
      except ValueError as e:
        print('Invalid command:',c)
    #print 'Position: %r'%(rq.status.gPO)
    print(rq.PrintStatus(rq.status))

  rq.Cleanup()

