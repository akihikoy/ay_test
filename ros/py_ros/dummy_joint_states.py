#!/usr/bin/python3
#\file    dummy_joint_states.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.09, 2017
import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg
import math

if __name__=='__main__':
  rospy.init_node('dummy_joint_states')
  pub_js= rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
  js= sensor_msgs.msg.JointState()
  js.name= rospy.get_param('controller_joint_names')
  js.header.seq= 0
  js.header.frame_id= ''
  t0= rospy.Time.now()
  while not rospy.is_shutdown():
    js.header.seq= js.header.seq+1
    js.header.stamp= rospy.Time.now()
    #js.position= [0.0]*7
    js.position= [1.5*math.sin(5.0*(rospy.Time.now()-t0).to_sec())]*len(js.name)
    js.velocity= [0.0]*len(js.name)
    js.effort= [0.0]*len(js.name)
    pub_js.publish(js)
    rospy.sleep(0.005)

