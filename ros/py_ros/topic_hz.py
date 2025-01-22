#!/usr/bin/python3
#\file    topic_hz.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.28, 2023
import roslib
import rospy
import rostopic
import sys

if __name__=='__main__':
  rospy.init_node('topic_hz')
  topic= sys.argv[1]
  hz= rostopic.ROSTopicHz(10)
  #msg_class,_,_= rostopic.get_topic_class(topic)
  sub= rospy.Subscriber(topic, rospy.AnyMsg, hz.callback_hz, callback_args=topic)
  while not rospy.is_shutdown():
    print(hz.get_hz(topic))
    #hz.print_hz([topic])
    #rospy.sleep(0.1)
    rospy.sleep(1.0)
