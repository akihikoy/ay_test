#!/usr/bin/python
#\file    velctrl3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.12, 2020
import roslib
import rospy
import std_msgs.msg
import controller_manager_msgs.srv
import math,time,sys

if __name__=='__main__':
  joint_idx= int(sys.argv[1]) if len(sys.argv)>1 else None
  rospy.init_node('velocity_control_test', disable_signals=True)  #NOTE: for executing the stop motion commands after Ctrl+C.

  pub= rospy.Publisher('/joint_group_vel_controller/command', std_msgs.msg.Float64MultiArray, queue_size=10)
  msg= std_msgs.msg.Float64MultiArray()
  msg.data= [0.0]*6
  msg.layout.data_offset= 1

  srv_sw_ctrl= rospy.ServiceProxy('/controller_manager/switch_controller', controller_manager_msgs.srv.SwitchController)
  sw_ctrl_req= controller_manager_msgs.srv.SwitchControllerRequest()
  sw_ctrl_req.strictness= sw_ctrl_req.STRICT
  sw_ctrl_req.stop_controllers= ['scaled_pos_joint_traj_controller']
  sw_ctrl_req.start_controllers= []
  srv_sw_ctrl(sw_ctrl_req)
  sw_ctrl_req.stop_controllers= []
  sw_ctrl_req.start_controllers= ['joint_group_vel_controller']
  srv_sw_ctrl(sw_ctrl_req)

  #rate= rospy.Rate(125)  #UR-CB receives velocities at 125 Hz.
  rate= rospy.Rate(500)  #UR-e receives velocities at 500 Hz.
  t0= time.time()

  try:
    while not rospy.is_shutdown():
      t= time.time()-t0
      angle= 0.05*math.sin(math.pi*t)
      if joint_idx is None:
        msg.data= [angle]*6
      else:
        msg.data= [0.0]*6
        msg.data[joint_idx]= angle
      pub.publish(msg)
      rate.sleep()

  except KeyboardInterrupt:
    print 'Interrupted'

  finally:
    print 'Stopping the robot...'
    msg.data= [0.0]*6
    pub.publish(msg)

    #WARNING: Without this loop or the sleep command, the robot moves slightly after switching the controller.
    for i in range(10):
      pub.publish(msg)
      rate.sleep()
    #rospy.sleep(0.05)

    sw_ctrl_req= controller_manager_msgs.srv.SwitchControllerRequest()
    sw_ctrl_req.strictness= sw_ctrl_req.STRICT
    sw_ctrl_req.stop_controllers= ['joint_group_vel_controller']
    sw_ctrl_req.start_controllers= []
    srv_sw_ctrl(sw_ctrl_req)
    sw_ctrl_req.stop_controllers= []
    sw_ctrl_req.start_controllers= ['scaled_pos_joint_traj_controller']
    srv_sw_ctrl(sw_ctrl_req)


