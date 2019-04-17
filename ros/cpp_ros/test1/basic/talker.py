#!/usr/bin/python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('rospy_tutorials')
import rospy
from std_msgs.msg import String

pub= rospy.Publisher('chatter',String)
rospy.init_node('talker',anonymous=True)
s= "hello world"
pub.publish(s)

