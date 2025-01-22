#!/usr/bin/python3
#\file    topic_available.py
#\brief   Function to check if a topic is available.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.15, 2024
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
import sensor_msgs.msg

def TopicArrives(port_name, port_type, time_out=5.0):
  try:
    res= rospy.wait_for_message(port_name, port_type, time_out)
    return res is not None
  except rospy.ROSException:
    return False

def IsTopicAvailable(port_name, port_type):
  try:
    topics= rospy.get_published_topics()
    for topic, topic_type in topics:
      if topic==port_name and topic_type==port_type._type:
        return True
    return False  # Topic not found or type mismatch
  except rospy.ROSException as e:
    #rospy.logerr("Failed to get published topics: %s", str(e))
    return False

if __name__=='__main__':
  rospy.init_node('topic_available')

  port_name= '/joint_states'
  port_type= sensor_msgs.msg.JointState

  print('Testing TopicArrives({},{})...'.format(port_name, port_type))
  print(' result:',TopicArrives(port_name, port_type))

  print('Testing IsTopicAvailable({},{})...'.format(port_name, port_type))
  print(' result:',IsTopicAvailable(port_name, port_type))

  port_name= '/joint_states2'

  print('Testing TopicArrives({},{})...'.format(port_name, port_type))
  print(' result:',TopicArrives(port_name, port_type))

  print('Testing IsTopicAvailable({},{})...'.format(port_name, port_type))
  print(' result:',IsTopicAvailable(port_name, port_type))

