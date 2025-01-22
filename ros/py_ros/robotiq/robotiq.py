#!/usr/bin/python3
#\file    robotiq.py
#\brief   Robotiq 2 finger gripper interface.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.24, 2015
'''
ref. swri-ros-pkg/robotiq/robotiq_c_model_control/nodes/CModelSimpleController.py
NOTE: run beforehand:
  $ ./RobotiqTcpCtrl.py
'''

import roslib
roslib.load_manifest('robotiq_c_model_control')
import rospy
import robotiq_c_model_control.msg as robotiq_msgs

#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

'''Robotiq Gripper utility class'''
class TRobotiq:
  def __init__(self):
    self.status= None
    self.SensorCallback= None

  '''Initialize (e.g. establish ROS connection).'''
  def Init(self, cmd_topic='/rq1/command', st_topic='/rq1/status'):
    self.pub_grip= rospy.Publisher(cmd_topic, robotiq_msgs.CModel_robot_output, queue_size=10)
    self.sub_grip= rospy.Subscriber(st_topic, robotiq_msgs.CModel_robot_input, self.SensorHandler)
    rospy.sleep(0.2)
    self.Activate()

  def Cleanup(self):
    #NOTE: cleaning-up order is important. consider dependency
    self.Deactivate()

  def SensorHandler(self,msg):
    self.status= msg
    if self.SensorCallback is not None:
      self.SensorCallback(self.status)

  @staticmethod
  def PrintStatus(st):
    print('Flags(ACT,GTO,STA,OBJ,FLT):',st.gACT,st.gGTO,st.gSTA,st.gOBJ,st.gFLT, end=' ')
    print('State(PR,PO,CU):',st.gPR,st.gPO,st.gCU)

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
    pos= max(0,min(255,int(pos)))
    cmd= robotiq_msgs.CModel_robot_output();
    cmd.rACT= 1
    cmd.rGTO= 1
    cmd.rPR= pos  #Position Request
    cmd.rSP= speed
    cmd.rFR= max_effort
    self.pub_grip.publish(cmd)
    if blocking:
      while pos!=self.status.gPR and not rospy.is_shutdown():
        #self.PrintStatus(self.status)
        rospy.sleep(0.001)
      prev_PO= None
      CMAX= 500
      counter= CMAX
      while not (self.status.gGTO==0 or self.status.gOBJ==3) and not rospy.is_shutdown():
        #self.PrintStatus(self.status)
        if self.status.gPO==prev_PO:  counter-= 1
        else:  counter= CMAX
        if counter==0:  break
        prev_PO= self.status.gPO
        rospy.sleep(0.001)
      #self.StopGripper()

  '''Stop the gripper motion. '''
  def StopGripper(self):
    cmd= robotiq_msgs.CModel_robot_output();
    cmd.rACT= 1
    cmd.rGTO= 0
    self.pub_grip.publish(cmd)


if __name__=='__main__':
  rospy.init_node('robotiq_test')
  rq= TRobotiq()
  rq.Init()
  rospy.sleep(1)
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
    if rq.status.gCU>2:
      l.event= 2
      rq.SensorCallback= None

  print('opening gripper')
  rq.MoveGripper(pos=0, max_effort=0, speed=255)
  print('closing gripper')
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

  rq.Cleanup()
