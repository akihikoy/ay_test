#!/usr/bin/env python
'''
ROS node for controling a Robotiq C-Model gripper using the Modbus TCP protocol.
ref. swri-ros-pkg/robotiq/robotiq_c_model_control/nodes/CModelTcpNode.py
'''

import roslib
roslib.load_manifest('robotiq_c_model_control')
roslib.load_manifest('robotiq_modbus_tcp')
import rospy
import robotiq_c_model_control.baseCModel
import robotiq_modbus_tcp.comModbusTcp
import os, sys
import robotiq_c_model_control.msg as robotiq_msgs

def MainLoop(address, node_name, sleep_time):
  #Gripper is a C-Model with a TCP connection
  gripper= robotiq_c_model_control.baseCModel.robotiqBaseCModel()
  gripper.client= robotiq_modbus_tcp.comModbusTcp.communication()
  #We connect to the address received as an argument
  gripper.client.connectToDevice(address)

  rospy.init_node(node_name)
  #Topic of gripper status
  status_pub= rospy.Publisher('~status', robotiq_msgs.CModel_robot_input, queue_size=10)
  #Topic of gripper command
  rospy.Subscriber('~command', robotiq_msgs.CModel_robot_output, gripper.refreshCommand)

  while not rospy.is_shutdown():
    #Get and publish the Gripper status
    status= gripper.getStatus()
    status_pub.publish(status)
    rospy.sleep(0.5*sleep_time)

    #Send the most recent command
    gripper.sendCommand()
    rospy.sleep(0.5*sleep_time)

if __name__ == '__main__':
  try:
    address= sys.argv[1] if len(sys.argv)>1 else 'rq1'
    node_name= sys.argv[2] if len(sys.argv)>2 else 'rq1'
    freq= float(sys.argv[3]) if len(sys.argv)>3 else 400.0
    MainLoop(address, node_name, 1.0/freq)
  except rospy.ROSInterruptException: pass
