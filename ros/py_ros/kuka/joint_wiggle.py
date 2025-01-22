#!/usr/bin/python3
#\file    joint_wiggle.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.08, 2017
import roslib; roslib.load_manifest('iiwa_ros')
import rospy
import iiwa_msgs.msg
import math

def QToKuka(q):
  msg= iiwa_msgs.msg.JointPosition()
  pos= msg.position
  joint_names= ('a%d'%d for d in range(1,8))
  for name,value in zip(joint_names,q):  setattr(pos,name,value)
  return msg

if __name__=='__main__':
  rospy.init_node('kuka_test')
  pub_joint= rospy.Publisher("/iiwa/command/JointPosition", iiwa_msgs.msg.JointPosition, queue_size=1)
  rate_adjuster= rospy.Rate(100)
  tm0= rospy.Time.now()
  while not rospy.is_shutdown():
    tm= (rospy.Time.now()-tm0).to_sec()
    q= [0.3*math.sin(2.0*tm) for d in range(7)]
    #print QToKuka(q)
    pub_joint.publish(QToKuka(q))
    rate_adjuster.sleep()
