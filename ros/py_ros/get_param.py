#!/usr/bin/python
#\file    get_param.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.06, 2017

import roslib; roslib.load_manifest('std_msgs')
import rospy
import sys

if __name__=='__main__':
  rospy.init_node('ros_min')
  param= rospy.get_param('~param', 'hoge')
  print 'param=',param

'''
#WARNING: The following code does not work.
#         We need to do init_node before using get_param.
if __name__=='__main__':
  param= rospy.get_param('~param', 'hoge')
  print 'param=',param
  rospy.init_node('ros_min')
'''
