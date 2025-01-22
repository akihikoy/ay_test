#!/usr/bin/python3
#\file    subscribe_any_topic.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.12, 2022
import roslib
import rospy
import rostopic
import sys

def Callback(msg):
  print('received:',type(msg))
  if hasattr(msg,'header'):  print(' ',msg.header.stamp.to_sec())

if __name__=='__main__':
  rospy.init_node('subscribe_any_topic')
  topic= sys.argv[1]
  msg_class,_,_= rostopic.get_topic_class(topic)
  print('topic:{}, msg_class:{}'.format(topic,msg_class))
  sub= rospy.Subscriber(topic, msg_class, Callback)
  rospy.spin()
