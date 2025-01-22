#!/usr/bin/python3
#\file    different_msg_pub.py
#\brief   Test: what if subscriber subscribes a topic
#         whose type is different from but has the same contents with
#         publisher's message.
#         NOTE: this works!
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.03, 2015
import roslib; roslib.load_manifest('ar_track_alvar')
import rospy
import ar_track_alvar.msg
import time

if __name__=='__main__':
  rospy.init_node('diff_msg_pub')
  pub_msg= rospy.Publisher('/the_topic', ar_track_alvar.msg.AlvarMarker)
  i= 1234
  while not rospy.is_shutdown():
    msg= ar_track_alvar.msg.AlvarMarker()
    msg.id= i
    print('send:',msg)
    pub_msg.publish(msg)
    time.sleep(0.5)
    i+= 1

