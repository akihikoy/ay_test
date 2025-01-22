#!/usr/bin/python3
#\file    ros_min.py
#\brief   Stand alone simple ROS test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.05, 2015
import roslib; roslib.load_manifest('std_msgs')
import rospy
import std_msgs.msg
import time

if __name__=='__main__':
  rospy.init_node('ros_min')
  pub_str= rospy.Publisher('/test', std_msgs.msg.String, queue_size=1)
  i= 0
  while not rospy.is_shutdown():
    a= (i*(i+1))/2
    s= 'ros_min ... %i'%a
    print('saying %r'%s)
    pub_str.publish(s)
    time.sleep(0.5)
    i+= 1

