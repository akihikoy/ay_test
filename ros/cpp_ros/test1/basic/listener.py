#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('rospy_tutorials')

import rospy
from std_msgs.msg import String

def callback(s):
  print "data is ...  "+s.data

rospy.init_node('listener',anonymous=True)

rospy.Subscriber("chatter",String,callback)
rospy.spin()

