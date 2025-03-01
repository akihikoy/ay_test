#!/usr/bin/python3
#\file    enable.py
#\brief   Enable Motoman.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.2, 2022
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv

if __name__=='__main__':
  rospy.init_node('motoman_test')
  rospy.wait_for_service('/robot_enable')
  srvp= rospy.ServiceProxy('/robot_enable', std_srvs.srv.Trigger)
  res= srvp()
  print('Ok' if res.success else 'Failed')
  print(res.message)
