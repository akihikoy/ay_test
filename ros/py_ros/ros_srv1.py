#!/usr/bin/python
#\file    ros_srv1.py
#\brief   Simple service server.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.30, 2015
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv

C= 1
def TestService(req):
  global C
  print 'Service required',C
  C+= 1
  return std_srvs.srv.EmptyResponse()

if __name__=='__main__':
  rospy.init_node('ros_srv1')
  s= rospy.Service('test_srv', std_srvs.srv.Empty, TestService)
  rospy.spin()
