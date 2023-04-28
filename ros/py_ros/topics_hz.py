#!/usr/bin/python
#\file    topics_hz.py
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
  topics= sys.argv[1:]
  hz= rostopic.ROSTopicHz(10)
  sub= dict()
  for topic in topics:
    sub[topic]= rospy.Subscriber(topic, rospy.AnyMsg, hz.callback_hz, callback_args=topic)
  def get_hz(t):
    v= hz.get_hz(t)
    return v[0] if v is not None else None
  while not rospy.is_shutdown():
    for topic in topics:
      print '{}: {}'.format(topic, get_hz(topic))
    #hz.print_hz([topic])
    #rospy.sleep(0.1)
    rospy.sleep(1.0)
    print '---'
