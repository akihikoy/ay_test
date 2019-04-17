#!/usr/bin/python
#\file    interactive1.py
#\brief   Interactive demo of Baymaxter
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015

import roslib
import rospy
import sensor_msgs.msg
import time
import math
import random

from bxtr import *
from state_machine import *
from baynat import GoNatural
from hug1 import DoHug
from hello1 import DoHello

State= dict()
State['someone_is_in_front']= None
State['some_people_around']= None
State['button_pushed']= None
def SetNone(key):
  State[key]= None

def IsPointInFront(points, max_angle, max_dist):
  for p in points:
    angle= math.atan2(p.y,p.x)
    dist= math.sqrt(p.x*p.x+p.y*p.y)
    #print (abs(angle),dist),
    if abs(angle)<max_angle and dist<max_dist:
      return True
  #print ''
  return False

def SonarCallBack(msg):
  if IsPointInFront(msg.points,30.0/180.0*math.pi,1.1):
    State['someone_is_in_front']= rospy.Time.now()
  else:
    if State['someone_is_in_front'] is not None and (rospy.Time.now()-State['someone_is_in_front']).to_sec()>0.5:
      State['someone_is_in_front']= None
  if IsPointInFront(msg.points,1.1*math.pi,2.0):
    State['some_people_around']= rospy.Time.now()
  else:
    if State['some_people_around'] is not None and (rospy.Time.now()-State['some_people_around']).to_sec()>2.0:
      State['some_people_around']= None

def NavCallback(value):
  if value:
    State['button_pushed']= rospy.Time.now()

#Generate a random number of uniform distribution of specified bound.
def Rand(xmin=-0.5,xmax=0.5):
  return random.random()*(xmax-xmin)+xmin

def PlayDice(prob):
  p= Rand(0.0,1.0)
  if p<=prob:  return True
  return False

if __name__=='__main__':
  rospy.init_node('baxter_test')

  sub_msg= rospy.Subscriber('/robot/sonar/head_sonar/state', sensor_msgs.msg.PointCloud, SonarCallBack)
  navigator_io= [None,None]
  navigator_io[RIGHT]= baxter_interface.Navigator(LRTostr(RIGHT))
  navigator_io[LEFT]= baxter_interface.Navigator(LRTostr(LEFT))
  # Navigator scroll wheel button press
  navigator_io[RIGHT].button0_changed.connect(NavCallback)
  navigator_io[LEFT].button0_changed.connect(NavCallback)

  #while not rospy.is_shutdown():
    #print State
    #print (rospy.Time.now()-State['someone_is_in_front']).to_sec() if State['someone_is_in_front'] is not None else None

  EnableBaxter()
  robot= TRobotBaxter()
  robot.Init()

  def InitMsg():
    print 'Hello. This is a Baymaxter interactive demo.'
    GoNatural(robot)

  sm= TStateMachine()
  sm.StartState= 'start'
  #sm.Debug= True

  sm['start']= TFSMState()
  sm['start'].EntryAction= InitMsg
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda: State['button_pushed'] is not None
  sm['start'].Actions[-1].Action= lambda: SetNone('button_pushed')
  sm['start'].Actions[-1].NextState= EXIT_STATE
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda: State['someone_is_in_front'] is not None
  sm['start'].Actions[-1].Action= lambda: DoHug(robot)
  sm['start'].Actions[-1].NextState= 'start'
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda: State['some_people_around'] is None and PlayDice(0.1)
  sm['start'].Actions[-1].Action= lambda: DoHello(robot)
  sm['start'].Actions[-1].NextState= 'start'
  sm['start'].NewAction()
  sm['start'].Actions[-1].Condition= lambda: State['some_people_around'] is None and PlayDice(0.1)
  sm['start'].Actions[-1].Action= lambda: DoHello(robot,True)
  sm['start'].Actions[-1].NextState= 'start'
  sm['start'].ElseAction.Condition= lambda: True
  sm['start'].ElseAction.Action= lambda: time.sleep(0.5)
  sm['start'].ElseAction.NextState= 'start'

  sm.Run()

  #rospy.spin()
  rospy.signal_shutdown('Done.')
