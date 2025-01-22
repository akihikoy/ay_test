#!/usr/bin/python3
#\file    service_available.py
#\brief   Function to check if a service is available.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.15, 2024
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
import sensor_msgs.msg
import rosgraph

def IsServiceAvailable(service_name):
  try:
    master= rosgraph.Master(rospy.get_name())
    service_uri= master.lookupService(service_name)
    return True
  except rosgraph.MasterError:
    return False
  except Exception as e:
    raise Exception('Error checking service availability: {}'.format(str(e)))

if __name__=='__main__':
  rospy.init_node('service_available')

  service_name= '/gripper_driver/move'

  print('Testing IsServiceAvailable({})...'.format(service_name))
  print(' result:',IsServiceAvailable(service_name))

  service_name= '/gripper_driver/jump'

  print('Testing IsServiceAvailable({})...'.format(service_name))
  print(' result:',IsServiceAvailable(service_name))

