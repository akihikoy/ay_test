#!/usr/bin/python
# -*- coding: utf-8 -*-

# execute:
#  rosrun nao_ctrl nao_walker.py --pip=163.221.139.138

import roslib
#roslib.load_manifest('nao_ctrl')
roslib.load_manifest('ros_sandbox')
import rospy
#from std_msgs.msg import String
from nao_msgs.msg import HeadAngles

pub= rospy.Publisher('head_angles',HeadAngles)
rospy.init_node('sender',anonymous=True)
a=HeadAngles()
a.yaw=0.5
a.pitch=-0.5
a.absolute=1
pub.publish(a)
