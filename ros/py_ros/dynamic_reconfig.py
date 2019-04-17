#!/usr/bin/python
#\file    dynamic_reconfig.py
#\brief   How to use dynamic_reconfigure client
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.04, 2016
'''
Here we want to do something like:
$ rosrun dynamic_reconfigure dynparam set /camera/driver data_skip 30
ref.
http://wiki.ros.org/hokuyo_node/Tutorials/UsingDynparamToChangeHokuyoLaserParameters#PythonAPI
'''
import roslib; roslib.load_manifest('std_msgs')
import rospy
import sys

import dynamic_reconfigure.client

if __name__=='__main__':
  rospy.init_node('ros_min')
  ds= int(sys.argv[1]) if len(sys.argv)>1 else 30
  try:
    client= dynamic_reconfigure.client.Client('/camera/driver',timeout=3.0)
  except rospy.exceptions.ROSException as e:
    print 'Error:',str(e)
    sys.exit()
  params= {'data_skip': ds}
  config= client.update_configuration(params)
