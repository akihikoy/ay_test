#!/usr/bin/python
#\file    get_q1.py
#\brief   Get current joint angles.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.12, 2018
#\version 0.2
#\date    Nov.21, 2019
import roslib
import rospy
import sensor_msgs.msg

joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

def GetState():
  try:
    state= rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, 5.0)
    q_map= {name:position for name,position in zip(state.name,state.position)}
    angles= [q_map[name] for name in joint_names]
    return state,angles
  except (rospy.ROSException, rospy.ROSInterruptException):
    raise Exception('Failed to read topic: /joint_states')

if __name__=='__main__':
  rospy.init_node('ur_test')

  for i in xrange(100000):
    if rospy.is_shutdown():  break

    x,q= GetState()
    print '@%d, x=%r'%(i,x)
    print '@%d, q(reordered)=%r'%(i,q)
    rospy.sleep(2.0e-3)
