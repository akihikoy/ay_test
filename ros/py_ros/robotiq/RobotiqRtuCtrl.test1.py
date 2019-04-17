#!/usr/bin/env python
'''
ROS node for controling a Robotiq C-Model gripper using the Modbus RTU protocol (serial via USB).
ref. /home/akihiko/catkin_ws/src/ros-industrial/robotiq/robotiq_c_model_control/nodes/CModelRtuNode.py
'''

import roslib; roslib.load_manifest('robotiq_c_model_control')
roslib.load_manifest('robotiq_modbus_rtu')
import rospy
import robotiq_c_model_control.baseCModel
import robotiq_modbus_rtu.comModbusRtu
import os, sys
import robotiq_c_model_control.msg as robotiq_msgs

from pymodbus.client.sync import ModbusSerialClient

import threading
locker= threading.RLock()

def CmdCallback(gripper,msg):
  gripper.refreshCommand(msg)
  with locker:
    gripper.sendCommand()
  rospy.sleep(0.005)

def MainLoop(device, node_name, timeout):
  #Gripper is a C-Model with an RTU connection
  gripper= robotiq_c_model_control.baseCModel.robotiqBaseCModel()
  gripper.client= robotiq_modbus_rtu.comModbusRtu.communication()

  #We connect to the address received as an argument
  #gripper.client.connectToDevice(device)
  #  We do not use gripper.client.connectToDevice since we want to control timeout.
  #timeout= 0.005  #0.2
  gripper.client.client= ModbusSerialClient(method='rtu',port=device,stopbits=1, bytesize=8, baudrate=115200, timeout=timeout)
  if not gripper.client.client.connect():
      print "Unable to connect to %s" % device

  rospy.init_node(node_name)
  #Topic of gripper status
  status_pub= rospy.Publisher('~status', robotiq_msgs.CModel_robot_input, queue_size=10)
  #Topic of gripper command
  #rospy.Subscriber('~command', robotiq_msgs.CModel_robot_output, gripper.refreshCommand)
  rospy.Subscriber('~command', robotiq_msgs.CModel_robot_output, lambda msg,g=gripper:CmdCallback(g,msg))

  while not rospy.is_shutdown():
    #Get and publish the Gripper status
    with locker:
      status= gripper.getStatus()
    status_pub.publish(status)
    rospy.sleep(0.005)

    #Send the most recent command
    #gripper.sendCommand()

if __name__ == '__main__':
  try:
    device= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyUSB0'
    node_name= sys.argv[2] if len(sys.argv)>2 else 'rq1'
    timeout= float(sys.argv[3]) if len(sys.argv)>3 else 0.005
    MainLoop(device, node_name, timeout)
  except rospy.ROSInterruptException: pass
