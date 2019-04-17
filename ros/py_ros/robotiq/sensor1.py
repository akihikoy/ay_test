#!/usr/bin/python
#\file    sensor1.py
#\brief   Robotiq Gripper sensing.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.23, 2015
'''
ref. swri-ros-pkg/robotiq/robotiq_c_model_control/nodes/CModelStatusListener.py
NOTE: run beforehand:
  xxx $ rosrun robotiq_c_model_control CModelTcpNode.py rq1
  $ ./RobotiqTcpCtrl.py
'''

import roslib
roslib.load_manifest('robotiq_c_model_control')
import rospy
import robotiq_c_model_control.msg as robotiq_msgs
import time, math

from grip1 import TRobotiq

def EchoSensor(msg):
  #print '---'
  #print msg
  print 'Flags(ACT,GTO,STA,OBJ,FLT):',msg.gACT,msg.gGTO,msg.gSTA,msg.gOBJ,msg.gFLT,
  print 'State(PR,PO,CU):',msg.gPR,msg.gPO,msg.gCU

if __name__=='__main__':
  rospy.init_node('robotiq_test')
  sub_grip= rospy.Subscriber('/rq1/status', robotiq_msgs.CModel_robot_input, EchoSensor)  #"CModelRobotInput"
  #rospy.spin()

  rq= TRobotiq()
  rq.Init()
  time.sleep(3)

  print 'closing gripper'
  #rq.CloseGripper()
  rq.MoveGripper(pos=255, max_effort=0, speed=0)
  time.sleep(5)
  print 'opening gripper'
  #rq.OpenGripper()
  rq.MoveGripper(pos=0, max_effort=0, speed=0)
  time.sleep(5)

  print 'Exit with Ctrl+C'
  rospy.spin()
