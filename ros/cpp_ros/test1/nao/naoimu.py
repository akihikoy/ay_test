#!/usr/bin/python
# -*- coding: utf-8 -*-

# execute:
#  rosrun nao_ctrl nao_sensors.py --pip=163.221.139.138

import roslib
#roslib.load_manifest('nao_ctrl')
roslib.load_manifest('ros_sandbox')

import rospy
#from std_msgs.msg import String
from nao_msgs.msg import TorsoIMU

def callback(v):
  print "data is ...  "
  print v

rospy.init_node('sensor',anonymous=True)

rospy.Subscriber("torso_imu",TorsoIMU,callback)
rospy.spin()

