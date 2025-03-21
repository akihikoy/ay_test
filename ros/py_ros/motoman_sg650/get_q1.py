#!/usr/bin/python3
#\file    get_q1.py
#\brief   Get current joint angles.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.2, 2022
import roslib; roslib.load_manifest('sensor_msgs')
import rospy
import sensor_msgs.msg

def GetState():
  try:
    state= rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, 5.0)
    return state
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to read topic: /joint_states')

if __name__=='__main__':
  rospy.init_node('motoman_test')

  for i in range(100000):
    if rospy.is_shutdown():  break

    x= GetState()
    q= x.position
    print('@%d, x=%r'%(i,x))
    print('@%d, q=%r'%(i,q))
    rospy.sleep(2.0e-3)
