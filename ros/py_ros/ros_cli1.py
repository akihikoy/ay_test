#!/usr/bin/python
#\file    ros_cli1.py
#\brief   Simple service server.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.30, 2015
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv

if __name__=='__main__':
  rospy.init_node('ros_cli1')
  rospy.wait_for_service('test_srv')
  srvp= rospy.ServiceProxy('test_srv', std_srvs.srv.Empty)
  for i in range(5):
    srvp()
