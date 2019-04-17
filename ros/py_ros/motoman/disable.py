#!/usr/bin/python
#\file    disable.py
#\brief   Disable Motoman.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv

if __name__=='__main__':
  rospy.init_node('motoman_test')
  rospy.wait_for_service('/robot_disable')
  srvp= rospy.ServiceProxy('/robot_disable', std_srvs.srv.Trigger)
  res= srvp()
  print 'Ok' if res.success else 'Failed'
  print res.message
