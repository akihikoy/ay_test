#!/usr/bin/python
#\file    grip1.py
#\brief   Robotiq Gripper control.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.23, 2015
'''
ref. swri-ros-pkg/robotiq/robotiq_c_model_control/nodes/CModelSimpleController.py
NOTE: run beforehand:
  xxx $ rosrun robotiq_c_model_control CModelTcpNode.py rq1
  $ ./RobotiqTcpCtrl.py
'''

import roslib
roslib.load_manifest('robotiq_c_model_control')
import rospy
#from robotiq_c_model_control.msg import _CModel_robot_output  as robotiq_out_msg
import robotiq_c_model_control.msg as robotiq_msgs
import time, math

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

'''Robotiq Gripper utility class'''
class TRobotiq:
  def __init__(self):
    pass

  '''Initialize (e.g. establish ROS connection).'''
  def Init(self, cmd_topic='/rq1/command'):  #cmd_topic='CModelRobotOutput'
    self.pub_grip= rospy.Publisher(cmd_topic, robotiq_msgs.CModel_robot_output)
    time.sleep(0.5)
    self.Activate()

  def Cleanup(self):
    #NOTE: cleaning-up order is important. consider dependency
    self.Deactivate()

  def Activate(self):
    cmd= robotiq_msgs.CModel_robot_output();
    cmd.rACT= 1
    cmd.rGTO= 1
    cmd.rSP= 255  #SPeed
    cmd.rFR= 150  #FoRce
    self.pub_grip.publish(cmd)

  def Deactivate(self):
    cmd= robotiq_msgs.CModel_robot_output();
    cmd.rACT= 0
    self.pub_grip.publish(cmd)

  '''Open a gripper.
    blocking: False: move background, True: wait until motion ends, 'time': wait until tN.  '''
  def OpenGripper(self, blocking=False):
    self.MoveGripper(pos=0, max_effort=100, blocking=blocking)

  '''Close a gripper.
    blocking: False: move background, True: wait until motion ends, 'time': wait until tN.  '''
  def CloseGripper(self, blocking=False):
    self.MoveGripper(pos=255, max_effort=100, blocking=blocking)

  '''Control a gripper.
    pos: target position; 0 (open), 255 (close).
    max_effort: maximum effort to control; 0~50 (weak), 200 (strong), 255 (maximum).
    speed: speed of the movement; 0 (minimum), 255 (maximum).
    blocking: False: move background, True: wait until motion ends, 'time': wait until tN.  '''
  def MoveGripper(self, pos, max_effort, speed=255, blocking=False):
    cmd= robotiq_msgs.CModel_robot_output();
    cmd.rACT= 1
    cmd.rGTO= 1
    cmd.rPR= pos  #Position Request
    cmd.rSP= speed
    cmd.rFR= max_effort
    self.pub_grip.publish(cmd)
    #TODO:FIXME:blocking is not implemented yet


if __name__=='__main__':
  rospy.init_node('robotiq_test')
  rq= TRobotiq()
  rq.Init()
  time.sleep(1)
  raw_input('wait activation>')
  print robotiq_msgs.CModel_robot_output()

  print 'closing gripper'
  rq.CloseGripper()
  time.sleep(1)
  print 'opening gripper'
  rq.OpenGripper()
  time.sleep(1)

  for t in FRange1(0.0,5.0,500):
    p= 255*0.5*(1.0-math.cos(t*2.0*math.pi/5.0))
    #print t,p
    rq.MoveGripper(p,100)
    time.sleep(0.01)

  rq.Cleanup()
