#!/usr/bin/python
#\file    get_q2.py
#\brief   Get joint angles.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.10, 2017
import roslib; roslib.load_manifest('motoman_driver')
import rospy
import sensor_msgs.msg

x_curr= None
q_curr= None
dq_curr= None

def JointStatesCallback(msg):
  global x_curr, q_curr, dq_curr
  x_curr= msg
  q_curr= x_curr.position
  dq_curr= x_curr.velocity
  #print 'x=%r'%(x_curr)

if __name__=='__main__':
  rospy.init_node('motoman_test')

  sub= rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, JointStatesCallback)

  for i in xrange(100000):
    if rospy.is_shutdown():  break

    print '@%d, x=%r'%(i,x_curr)
    print '@%d, q=%r'%(i,q_curr)
    print '@%d, dq=%r'%(i,dq_curr)
    rospy.sleep(2.0e-3)
