#!/usr/bin/python
#\file    rosparam_1.py
#\brief   Test of uploading to / downloading from the ROS parameter server.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.19, 2023
'''
./rosparam_1.py _role:=up
./rosparam_1.py _role:=down
'''
from __future__ import print_function
import roslib; roslib.load_manifest('std_msgs')
import rospy
import sys
import numpy as np

if __name__=='__main__':
  node_name= 'ros_param_{}'.format(np.random.randint(0,100))
  rospy.init_node(node_name)
  role= rospy.get_param('~role', 'up')  #Role of this node; up or down.
  print('{}: role: {}'.format(node_name, role))

  keys= ('param1_int','param2_float','param3_array','param4_str', 'param5_float2')
  if role=='up':
    rospy.set_param('/test_category/param1_int', 125)
    rospy.set_param('/test_category/param2_float', 1.2345)
    rospy.set_param('/test_category/param3_array', (np.array(list(range(10)))*0.1).tolist())
    rospy.set_param('/test_category/param4_str', 'hoge hoge')
    rospy.set_param('/test_category/param5_float2', float(12345))
    print('Uploaded param: {}'.format(keys))
  elif role=='down':
    for key in keys:
      p= rospy.get_param('/test_category/{}'.format(key), None)
      print('Get param /test_category/{}: value: {} type: {}'.format(key,p,type(p)))
  else:
    raise Exception('Invalid role: {}'.format(role))
