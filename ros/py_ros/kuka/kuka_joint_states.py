#!/usr/bin/python
#\file    kuka_joint_states.py
#\brief   Convert /iiwa/state/JointPosition topic to /joint_states.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.08, 2017
import roslib; roslib.load_manifest('iiwa_ros')
import rospy
import sensor_msgs.msg
import iiwa_msgs.msg
import copy

def Callback(pub_st, msg):
  #remove_names= ('head_nod','torso_t0')
  #if all(rmname not in msg.name for rmname in remove_names):
    #pub_st.publish(msg)
  #else:
    #msg2= sensor_msgs.msg.JointState()
    #msg2.header= msg.header
    #idxs= [i for i,name in enumerate(msg.name) if name not in remove_names]
    #msg2.name= [msg.name[i] for i in idxs]
    #if len(msg.position)>0:  msg2.position= [msg.position[i] for i in idxs]
    #if len(msg.velocity)>0:  msg2.velocity= [msg.velocity[i] for i in idxs]
    #if len(msg.effort)>0:  msg2.effort= [msg.effort[i] for i in idxs]
    #pub_st.publish(msg2)
  joint_names= ('a%d'%d for d in xrange(1,8))
  msg2= sensor_msgs.msg.JointState()
  msg2.header= msg.header
  #idxs= [i for i,name in enumerate(msg.name) if name not in remove_names]
  msg2.name= ['iiwa_joint_%d'%d for d in xrange(1,8)]
  msg2.position= [getattr(msg.position,name) for name in joint_names]
  #if len(msg.velocity)>0:  msg2.velocity= [msg.velocity[i] for i in idxs]
  #if len(msg.effort)>0:  msg2.effort= [msg.effort[i] for i in idxs]
  pub_st.publish(msg2)

if __name__=='__main__':
  rospy.init_node('kuka_joint_states')
  pub_st= rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
  sub_st= rospy.Subscriber('/iiwa/state/JointPosition', iiwa_msgs.msg.JointPosition, lambda msg: Callback(pub_st,msg))
  rospy.spin()
