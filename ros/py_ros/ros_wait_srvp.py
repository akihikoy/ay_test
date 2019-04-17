#!/usr/bin/python
#\file    ros_wait_srvp.py
#\brief   Test wait service.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.30, 2015
import roslib; roslib.load_manifest('std_srvs')
import rospy
import std_srvs.srv
#import rospy_tutorials.srv

def SetupServiceProxy(name, srv_type, persistent=False, time_out=None):
  print 'Waiting for %s... (t/o: %r)' % (name, time_out)
  try:
    rospy.wait_for_service(name, time_out)
  except rospy.exceptions.ROSException as e:
    print 'Failed to connect the service %s' % name
    print '  Error:',str(e)
    return None
  srvp= rospy.ServiceProxy(name, srv_type, persistent=persistent)
  return srvp

if __name__=='__main__':
  rospy.init_node('ros_wait_srvp')

  servp= SetupServiceProxy('/test_srv', std_srvs.srv.Empty, time_out=1.0)

  '''NOTE: This (service type is wrong) works, but when using servp()
    it fails with an exception:
      ospy.service.ServiceException: unable to connect to service: remote error reported: request from [/ros_wait_srvp]: md5sums do not match: [6a2e34150c00229791cc89ff309fff21] vs. [d41d8cd98f00b204e9800998ecf8427e]'''
  #servp= SetupServiceProxy('/test_srv', rospy_tutorials.srv.AddTwoInts, time_out=1.0)

  print 'Got the service:',servp
  if servp!=None:  servp()

