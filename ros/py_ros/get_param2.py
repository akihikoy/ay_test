#!/usr/bin/python3
#\file    get_param2.py
#\brief   Test of get_param with a complicated type (dict).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.18, 2022

'''
Examples
$ ./get_param2.py _param:='{}'
param= {}
$ ./get_param2.py _param:='{"a":100}'
param= {'a': 100}
$ ./get_param2.py _param:='{"a":100,"b":200}'
param= {'a': 100, 'b': 200}
$ ./get_param2.py _param:='{"a":100,"b":200,"c":{"d":10}}'
param= {'a': 100, 'c': {'d': 10}, 'b': 200}
'''

import roslib; roslib.load_manifest('std_msgs')
import rospy
import sys

if __name__=='__main__':
  rospy.init_node('ros_param')
  param= rospy.get_param('~param', {'a':100,'b':200})
  print('param=',param)
