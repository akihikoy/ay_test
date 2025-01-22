#!/usr/bin/python3
#\file    ros_sub1.py
#\brief   Stand alone simple ROS test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
import roslib; roslib.load_manifest('std_msgs')
import rospy
import std_msgs.msg
import time

def Callback(msg):
  print('received:',msg)

if __name__=='__main__':
  rospy.init_node('ros_sub1')
  sub= rospy.Subscriber('/test', std_msgs.msg.String, Callback)
  rospy.spin()
