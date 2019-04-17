#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('ros_sandbox')
import rospy
from ros_sandbox.msg import Mtest

def callback(a):
  print "data is ...  "
  print a

rospy.init_node('listener2',anonymous=True)

rospy.Subscriber("chatter2",Mtest,callback)
rospy.spin()

