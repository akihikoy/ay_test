#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('ros_sandbox')
#roslib.load_manifest('rospy_tutorials')
#  DO NOT use multiple load_manifest calls. 
#  If you need multiple calls, it's probably because you're
#  missing the correct dependencies in your manifest.
import rospy
from ros_sandbox.msg import Mtest

pub= rospy.Publisher('chatter2',Mtest)
rospy.init_node('talker2',anonymous=True)

while not rospy.is_shutdown():
  a=Mtest()
  a.x= 2.5
  a.y= -1.2
  print a
  pub.publish(a)
