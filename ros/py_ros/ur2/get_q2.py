#!/usr/bin/python3
#\file    get_q2.py
#\brief   Get joint angles.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.12, 2018
#\version 0.2
#\date    Nov.21, 2019
import roslib
import rospy
import sensor_msgs.msg

x_curr= None
q_curr= None
dq_curr= None
joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

def JointStatesCallback(msg):
  global x_curr, q_curr, dq_curr
  x_curr= msg
  q_map= {name:position for name,position in zip(x_curr.name,x_curr.position)}
  dq_map= {name:velocity for name,velocity in zip(x_curr.name,x_curr.velocity)}
  q_curr= [q_map[name] for name in joint_names]
  dq_curr= [dq_map[name] for name in joint_names]
  #print 'x=%r'%(x_curr)

if __name__=='__main__':
  rospy.init_node('ur_test')

  sub= rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, JointStatesCallback)

  for i in range(100000):
    if rospy.is_shutdown():  break

    print('@%d, x=%r'%(i,x_curr))
    print('@%d, q(reordered)=%r'%(i,q_curr))
    print('@%d, dq(reordered)=%r'%(i,dq_curr))
    rospy.sleep(2.0e-3)
