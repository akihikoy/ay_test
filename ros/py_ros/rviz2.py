#!/usr/bin/python
#\file    rviz2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.25, 2019
import roslib; roslib.load_manifest('std_msgs')
import rospy
import tf
import visualization_msgs.msg
import geometry_msgs.msg
import math

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

if __name__=='__main__':
  rospy.init_node('ros_min')
  viz_pub= rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=1)

  t= rospy.Time.now()
  while not rospy.is_shutdown():
    marker= visualization_msgs.msg.Marker()
    marker.header.frame_id= 'base_link'
    marker.header.stamp= rospy.Time.now()
    marker.ns= 'visualizer'
    marker.id= 0
    marker.action= visualization_msgs.msg.Marker.ADD  # or DELETE
    marker.lifetime= rospy.Duration(1.0)
    marker.type= visualization_msgs.msg.Marker.CUBE  # or CUBE, SPHERE, ARROW, CYLINDER
    marker.scale.x= 0.2
    marker.scale.y= 0.2
    marker.scale.z= 0.2
    marker.color.a= 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.pose= XToGPose([0.0,0.0,0.0, 0.0,0.0,0.0,1.0])
    marker.pose.position.x= 1.0*math.sin((rospy.Time.now()-t).to_sec())
    viz_pub.publish(marker)
    rospy.sleep(0.05)

  viz_pub.publish()
  viz_pub.unregister()
