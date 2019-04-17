#!/usr/bin/python
#\file    joint_spring2.py
#\brief   joint spring test
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.12, 2015

import rospy
from bxtr import BxJointSprings
RIGHT=0
LEFT=1

if __name__=='__main__':
  rospy.init_node('baxter_test')

  bxjs= BxJointSprings()  #virtual joint spring controller
  bxjs.AttachSprings(arms=(RIGHT,LEFT), stop_err=0.2, stop_dt=None)

  rospy.signal_shutdown('Done.')
