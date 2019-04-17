#!/usr/bin/python
# -*- coding: utf-8 -*-

# execute:
#  rosrun nao_ctrl nao_walker.py --pip=163.221.139.138

import roslib
#roslib.load_manifest('nao_ctrl')
roslib.load_manifest('ros_sandbox')
import rospy
from std_msgs.msg import String

pub= rospy.Publisher('speech',String)
rospy.init_node('speaker',anonymous=True)
s= "hello world xxx"
pub.publish(s)

