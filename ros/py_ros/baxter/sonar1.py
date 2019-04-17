#!/usr/bin/python
#\file    sonar1.py
#\brief   Baxter: getting data from sonor sensor
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2015

import roslib
import rospy
import sensor_msgs.msg
import baxter_interface
import time
import math

def IsPointInFront(points, max_angle, max_dist):
  for p in points:
    angle= math.atan2(p.y,p.x)
    dist= math.sqrt(p.x*p.x+p.y*p.y)
    #print (abs(angle),dist),
    if abs(angle)<max_angle and dist<max_dist:
      return True
  #print ''
  return False

def CallBack(msg):
  #print '----------------'
  #print msg
  if IsPointInFront(msg.points,30.0/180.0*math.pi,1.1):  print 'Found a near point!',msg.header.seq

if __name__=='__main__':
  rospy.init_node('baxter_test')

  sub_msg= rospy.Subscriber('/robot/sonar/head_sonar/state', sensor_msgs.msg.PointCloud, CallBack)

  rospy.spin()
  #rospy.signal_shutdown('Done.')
