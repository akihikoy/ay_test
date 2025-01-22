#!/usr/bin/python3
#\file    fragile.py
#\brief   Grasping a fragile object.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.11, 2015

import roslib
import rospy
from robotiq import TRobotiq

if __name__=='__main__':
  rospy.init_node('robotiq_test')
  rq= TRobotiq()
  rq.Init()
  #rospy.sleep(1)
  input('wait activation>')

  #def sensor_callback(st):
    #rq.PrintStatus(st)
  #rq.SensorCallback= sensor_callback
  #print 'opening gripper'
  #rq.MoveGripper(pos=0, max_effort=0, speed=255, blocking=True)
  #print 'closing gripper'
  #rq.MoveGripper(pos=255, max_effort=0, speed=0, blocking=True)

  class TContainer:
    pass
  l= TContainer()
  l.event= 0
  def sensor_callback(st):
    rq.PrintStatus(st)
    if rq.status.gGTO==0 or rq.status.gOBJ==3:
      l.event= 1
      rq.SensorCallback= None
    if rq.status.gCU>2:  #2 IS A THRESHOLD (SMALLER IS MORE SENSITIVE)
      l.event= 2
      rq.SensorCallback= None

  print('opening gripper')
  rq.MoveGripper(pos=0, max_effort=255, speed=255)
  #print 'closing gripper'
  input('start closing?>')

  rq.MoveGripper(pos=255, max_effort=0, speed=0)
  while rq.status.gPR!=255:  rospy.sleep(0.001)
  rq.SensorCallback= sensor_callback
  while not rospy.is_shutdown():
    if l.event==1:
      break
    if l.event==2:
      rq.StopGripper()
      break
    rospy.sleep(0.001)
  print('event=',l.event)

  input('opening gripper?>')
  rq.MoveGripper(pos=0, max_effort=255, speed=255, blocking=True)
  rospy.sleep(1)

  rq.Cleanup()
