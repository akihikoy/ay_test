#!/usr/bin/python3
#\file    ros_sub_multiple.py
#\brief   ROS subscriber test of subscribing the same topic in multiple subscribers in the same program.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
import roslib; roslib.load_manifest('std_msgs')
import rospy
import std_msgs.msg
import time

def Callback1(msg):
  print('Receiver1:',msg)

def Callback2(msg):
  print('Receiver2:',msg)

def Callback3(msg):
  print('Receiver3:',msg)

if __name__=='__main__':
  rospy.init_node('ros_sub_multiple')
  sub= rospy.Subscriber('/test', std_msgs.msg.String, Callback1)
  sub= rospy.Subscriber('/test', std_msgs.msg.String, Callback2)
  sub= rospy.Subscriber('/test', std_msgs.msg.String, Callback3)
  rospy.spin()
