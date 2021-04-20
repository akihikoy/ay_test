#!/usr/bin/python
#\file    send_fake_io1.py
#\brief   Sending a fake IO states (digital in).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.20, 2021
import roslib
import rospy
import std_msgs.msg
import std_srvs.srv
#roslib.load_manifest('ur_dashboard_msgs')
#import ur_dashboard_msgs.msg
roslib.load_manifest('ur_msgs')
import ur_msgs.msg

def SendFakeDigitalInDignal(signal_idx, signal_trg):
  pub_io_states= rospy.Publisher('/ur_hardware_interface/io_states', ur_msgs.msg.IOStates, queue_size=10)
  rospy.sleep(0.2)

  msg= ur_msgs.msg.IOStates()

  msg.digital_in_states= [ur_msgs.msg.Digital(pin,False) for pin in range(18)]
  msg.digital_out_states= [ur_msgs.msg.Digital(pin,False) for pin in range(18)]
  msg.flag_states= [ur_msgs.msg.Digital(pin,False) for pin in range(2)]
  msg.analog_in_states= [ur_msgs.msg.Analog(pin,0,0) for pin in range(2)]
  msg.analog_out_states= [ur_msgs.msg.Analog(pin,0,0) for pin in range(2)]

  msg.digital_in_states[signal_idx]= ur_msgs.msg.Digital(signal_idx,signal_trg)
  #print 'msg='
  #print msg
  pub_io_states.publish(msg)

  #pub_digital= rospy.Publisher('/ur_hardware_interface/io_states', ur_msgs.msg.Digital, queue_size=10)
  #msg= ur_msgs.msg.Digital(2,True)
  #print 'msg='
  #print msg
  #pub_digital.publish(msg)

if __name__=='__main__':
  rospy.init_node('send_fake_io1')
  SendFakeDigitalInDignal(3, True)
  #rate= rospy.Rate(10)
  #for i in range(3):
    #SendFakeDigitalInDignal(3, True)
    #print '=============',i,'============='
    #rate.sleep()
  ##rospy.spin()
