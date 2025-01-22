#!/usr/bin/python3
#\file    get_q3.py
#\brief   Get joint angles with Gen3's service.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.10, 2019
import roslib
import rospy
import kortex_driver.srv
import kortex_driver.msg

if __name__=='__main__':
  rospy.init_node('gen3_test')
  rospy.wait_for_service('RefreshFeedback')
  srvRefreshFeedback= rospy.ServiceProxy('RefreshFeedback', kortex_driver.srv.RefreshFeedback)

  feedback= srvRefreshFeedback()
  print(feedback)
  q= [a.position/180.0*3.141592653589793 for a in feedback.output.actuators]
  print('q=',q)
