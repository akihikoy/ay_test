#!/usr/bin/python3
#\file    get_x1.py
#\brief   Compare the Cartesian pose values: KDL estimate vs. base_feedback.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019
'''
cf. https://github.com/Kinovarobotics/ros_kortex/issues/26
* Only positional component changes.
* x_js-x_bf does not change when moving the robot linearly in Cartesian space.
  -0.1114 -0.0445  0.0014
* x_js-x_bf changes when moving the robot rotationally in Cartesian space.
  -0.0952 -0.0713 -0.0157
'''

import roslib
import rospy
import tf
import sensor_msgs.msg
import kortex_driver.msg

import numpy as np
from kdl_kin2 import TKinematics

import visualization_msgs.msg
import geometry_msgs.msg

#Convert x to geometry_msgs/Pose
def XToGPose(x):
  pose= geometry_msgs.msg.Pose()
  pose.position.x= x[0]
  pose.position.y= x[1]
  pose.position.z= x[2]
  pose.orientation.x= x[3]
  pose.orientation.y= x[4]
  pose.orientation.z= x[5]
  pose.orientation.w= x[6]
  return pose

viz_pub= None

def VizCube(id,x,rgb,scale=0.01):
  global viz_pub
  marker= visualization_msgs.msg.Marker()
  marker.header.frame_id= 'base_link'
  marker.header.stamp= rospy.Time.now()
  marker.ns= 'visualizer'
  marker.id= id
  marker.action= visualization_msgs.msg.Marker.ADD  # or DELETE
  marker.lifetime= rospy.Duration(1.0)
  marker.type= visualization_msgs.msg.Marker.CUBE  # or CUBE, SPHERE, ARROW, CYLINDER
  marker.scale.x= scale
  marker.scale.y= scale
  marker.scale.z= scale
  marker.color.a= 1.0
  marker.color.r = rgb[0]
  marker.color.g = rgb[1]
  marker.color.b = rgb[2]
  marker.pose= XToGPose(x)
  viz_pub.publish(marker)


kin= None
x_js= None

def JointStatesCallback(msg):
  global x_js
  q0= msg.position
  angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x_js= kin.forward_position_kinematics(angles)
  VizCube(0,x_js,[1.,0.,0.])
  #print 'q0=',np.array(q0)
  #print 'x0= FK(q0)=',x0

def BaseFeedbackCallback(feedback):
  global x_js
  deg2rad= lambda q:q/180.0*np.pi
  x= feedback.base.tool_pose_x
  y= feedback.base.tool_pose_y
  z= feedback.base.tool_pose_z
  theta_x= deg2rad(feedback.base.tool_pose_theta_x)
  theta_y= deg2rad(feedback.base.tool_pose_theta_y)
  theta_z= deg2rad(feedback.base.tool_pose_theta_z)
  Q= tf.transformations.quaternion_from_euler(theta_x,theta_y,theta_z)
  x_bf= np.array([x,y,z]+list(Q))
  VizCube(1,x_bf,[0.,0.,1.])
  print('x_js-x_bf=', np.array([v if abs(v)>1.0e-6 else 0 for v in (x_js-x_bf)[:3]]+[v if abs(v)>1.0e-3 else 0 for v in (x_js-x_bf)[3:]]))

if __name__=='__main__':
  np.set_printoptions(precision=4)
  rospy.init_node('gen3_test')

  viz_pub= rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=1)

  kin= TKinematics(base_link='base_link',end_link='end_effector_link',description='/gen3a/robot_description')

  subjs= rospy.Subscriber('/gen3a/joint_states', sensor_msgs.msg.JointState, JointStatesCallback)
  subbf= rospy.Subscriber('/gen3a/base_feedback', kortex_driver.msg.BaseCyclic_Feedback, BaseFeedbackCallback)

  rospy.spin()

