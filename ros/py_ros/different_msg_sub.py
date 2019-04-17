#!/usr/bin/python
#\file    different_msg_sub.py
#\brief   Test: what if subscriber subscribes a topic
#         whose type is different from but has the same contents with
#         publisher's message.
#         NOTE: this works!
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.03, 2015
import roslib; roslib.load_manifest('ar_track_alvar_msgs')
import rospy
import ar_track_alvar_msgs.msg
import time

def CallBack(msg):
  print 'get:',msg

if __name__=='__main__':
  rospy.init_node('ros_min')
  sub_msg= rospy.Subscriber('/the_topic', ar_track_alvar_msgs.msg.AlvarMarker, CallBack)
  rospy.spin()
