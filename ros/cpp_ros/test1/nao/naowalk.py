#!/usr/bin/python
# -*- coding: utf-8 -*-

# execute:
#  rosrun nao_ctrl nao_walker.py --pip=163.221.139.138

import roslib
#roslib.load_manifest('nao_ctrl')
roslib.load_manifest('ros_sandbox')
import rospy
import time
#from std_msgs.msg import String
from nao_msgs.msg import HeadAngles
from geometry_msgs.msg import Twist

pub= rospy.Publisher('cmd_vel',Twist)
rospy.init_node('sender',anonymous=True)
v=Twist()
#v.linear.x=0.5
v.angular.z=0.5
pub.publish(v)

time.sleep(2)

pub.publish(Twist())

