#!/usr/bin/python3
#\file    thread_wait_ros_srv1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.26, 2021
import roslib
import sys
import threading
import rospy
import std_srvs.srv

def SetupServiceProxy(name, srv_type, persistent=False, time_out=None):
  print('Waiting for %s... (t/o: %r)' % (name, time_out))
  try:
    rospy.wait_for_service(name, time_out)
  except rospy.exceptions.ROSException as e:
    print('Failed to connect the service %s' % name)
    print('  Error:',str(e))
    return None
  srvp= rospy.ServiceProxy(name, srv_type, persistent=persistent)
  return srvp

if __name__=='__main__':
  services= ['power_on', 'power_off', 'brake_release', 'play', 'stop', 'shutdown', 'unlock_protective_stop', '', '']

  #We want to execute them in parallel:
  #for service in services:
    #srvp_ur_dashboard[service]= SetupServiceProxy('/ur_hardware_interface/dashboard/{0}'.format(service), std_srvs.srv.Trigger, persistent=False, time_out=3.0)

  threads= {}
  srvp_ur_dashboard= {}
  for service in services:
    threads[service]= threading.Thread(name=service, target=lambda:(srvp_ur_dashboard.__setitem__(service,SetupServiceProxy('/ur_hardware_interface/dashboard/{0}'.format(service), std_srvs.srv.Trigger, persistent=False, time_out=3.0))))
    threads[service].start()
  for service,th in threads.items():  th.join()

