#!/usr/bin/python
#\file    enable.py
#\brief   Enable Motoman.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv

if __name__=='__main__':
  rospy.init_node('motoman_test')
  rospy.wait_for_service('/robot_enable')
  srvp= rospy.ServiceProxy('/robot_enable', std_srvs.srv.Trigger)
  res= srvp()
  print 'Ok' if res.success else 'Failed'
  print res.message
