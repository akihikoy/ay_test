#!/usr/bin/python3
#\file    keyctrl1.py
#\brief   Keyboard control to test switching the trajectory and velocity controllers.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.14, 2020

import roslib
import rospy
import actionlib
import sensor_msgs.msg
import control_msgs.msg
import trajectory_msgs.msg
import std_msgs.msg
import controller_manager_msgs.srv
import time, math, sys, copy
from kdl_kin2 import TKinematics
from kbhit2 import TKBHit
import threading
import numpy as np

class TUR(object):
  def __init__(self):
    self.joint_names= ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                      'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    self.x_curr= None
    self.q_curr= [0.0]*6
    self.dq_curr= [0.0]*6

    self.kin= TKinematics(base_link='base_link',end_link='tool0')

    self.traj_client= actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
    # Wait some seconds for the head action server to start or exit
    if not self.traj_client.wait_for_server(rospy.Duration(5.0)):
      rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
      rospy.signal_shutdown('Action Server not found')
      sys.exit(1)

    self.vel_pub= rospy.Publisher('/joint_group_vel_controller/command', std_msgs.msg.Float64MultiArray, queue_size=10)

    self.srv_sw_ctrl= rospy.ServiceProxy('/controller_manager/switch_controller', controller_manager_msgs.srv.SwitchController)

    self.js_sub= rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, self.JointStatesCallback)
    rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, 5.0)

  def Q(self):
    return self.q_curr

  def J(self, q):
    angles= {joint:q[j] for j,joint in enumerate(self.joint_names)}  #Deserialize
    J_res= self.kin.jacobian(joint_values=angles)
    return J_res

  def JointStatesCallback(self, msg):
    self.x_curr= msg
    q_map= {name:position for name,position in zip(self.x_curr.name,self.x_curr.position)}
    dq_map= {name:velocity for name,velocity in zip(self.x_curr.name,self.x_curr.velocity)}
    self.q_curr= [q_map[name] for name in self.joint_names]
    self.dq_curr= [dq_map[name] for name in self.joint_names]

  def MoveTo(self, q, dt):
    self.traj_client.cancel_goal()  #Ensure to cancel the ongoing goal.

    goal= control_msgs.msg.FollowJointTrajectoryGoal()
    #goal.goal_time_tolerance= rospy.Time(0.1)
    goal.trajectory.joint_names= self.joint_names
    #NOTE: We need to specify velocities. Otherwise:
    #error_code: -1
    #error_string: "Received a goal without velocities"
    def add_point(goal, time, positions, velocities):
      point= trajectory_msgs.msg.JointTrajectoryPoint()
      point.positions= copy.deepcopy(positions)
      point.velocities= copy.deepcopy(velocities)
      point.time_from_start= rospy.Duration(time)
      goal.trajectory.points.append(point)

    q0= self.Q()
    add_point(goal, 0.0, q0, [0.0]*6)
    add_point(goal, dt, q, [0.0]*6)

    goal.trajectory.header.stamp= rospy.Time.now()
    self.traj_client.send_goal(goal)
    #self.traj_client.cancel_goal()
    #self.traj_client.wait_for_result(timeout=rospy.Duration(20.0))

    self.traj_client.wait_for_result()
    print(self.traj_client.get_result())

  class VelCtrl(object):
    def __init__(self,ur):
      self.ur= ur
      self.msg= std_msgs.msg.Float64MultiArray()
      self.msg.data= [0.0]*6
      self.msg.layout.data_offset= 1
      self.mode= 'traj'
      self.StartVelCtrlMode()

    def __del__(self):
      self.StopVelCtrlMode()

    def __enter__(self, *args, **kwargs):
      return self

    def __exit__(self, *args, **kwargs):
      self.StopVelCtrlMode()

    def StartVelCtrlMode(self):
      if self.mode=='vel':  return
      sw_ctrl_req= controller_manager_msgs.srv.SwitchControllerRequest()
      sw_ctrl_req.strictness= sw_ctrl_req.STRICT
      sw_ctrl_req.stop_controllers= ['scaled_pos_joint_traj_controller']
      sw_ctrl_req.start_controllers= []
      self.ur.srv_sw_ctrl(sw_ctrl_req)
      sw_ctrl_req.stop_controllers= []
      sw_ctrl_req.start_controllers= ['joint_group_vel_controller']
      self.ur.srv_sw_ctrl(sw_ctrl_req)
      self.mode= 'vel'

    def StopVelCtrlMode(self):
      if self.mode=='traj':  return
      self.Step([0.0]*6)
      sw_ctrl_req= controller_manager_msgs.srv.SwitchControllerRequest()
      sw_ctrl_req.strictness= sw_ctrl_req.STRICT
      sw_ctrl_req.stop_controllers= ['joint_group_vel_controller']
      sw_ctrl_req.start_controllers= []
      self.ur.srv_sw_ctrl(sw_ctrl_req)
      sw_ctrl_req.stop_controllers= []
      sw_ctrl_req.start_controllers= ['scaled_pos_joint_traj_controller']
      self.ur.srv_sw_ctrl(sw_ctrl_req)
      self.mode= 'traj'

    def Step(self,dq):
      if self.mode!='vel':
        print('Warning: The velocity control mode is not active.')
        return
      self.msg.data= dq
      self.ur.vel_pub.publish(self.msg)


def ReadKeyboard(is_running, key_cmd, key_locker):
  with TKBHit() as kbhit:
    dt_hold= 0.2
    t_prev= 0
    while is_running[0]:
      c= kbhit.KBHit()
      if c is not None or time.time()-t_prev>dt_hold:
        with key_locker:
          key_cmd[0]= c
        t_prev= time.time()
      time.sleep(0.0025)


if __name__=='__main__':
  rospy.init_node('ur_key_control', disable_signals=True)  #NOTE: for executing the stop motion commands after Ctrl+C.

  ur= TUR()
  q_start= ur.Q()
  print(q_start)

  #KBD thread
  key_cmd= [None]
  key_locker= threading.RLock()
  is_running= [True]
  t1= threading.Thread(name='t1', target=lambda a1=is_running,a2=key_cmd,a3=key_locker: ReadKeyboard(a1,a2,a3))
  t1.start()

  key_map= {'s':'+x', 'x':'-x', 'z':'+y', 'c':'-y', 'd':'+z', 'a':'-z'}
  pmov_map= {'+x':(0,+1.0), '-x':(0,-1.0), '+y':(1,+1.0), '-y':(1,-1.0), '+z':(2,+1.0), '-z':(2,-1.0)}

  ctrl_hz= 125  #UR-CB.
  #ctrl_hz= 500  #UR-CB.
  rate= rospy.Rate(ctrl_hz)
  dt= 1.0/ctrl_hz

  speed_max= 0.06  #Max speed.
  acceleration= 0.5
  dp_curr= [0.0, 0.0, 0.0]  #Current velocity.

  try:
    with ur.VelCtrl(ur) as velctrl:
      while not rospy.is_shutdown():
        with key_locker:
          c= key_cmd[0];
        mov= None
        if c is not None:
          if c=='q':  break
          elif c in list(key_map.keys()):  mov= key_map[c]
          elif c==' ':  mov= 'move_to_start'

        if mov=='move_to_start':
          velctrl.StopVelCtrlMode()
          ur.MoveTo(q_start, 2.0)
          velctrl.StartVelCtrlMode()
          continue

        elif mov in list(pmov_map.keys()):
          ixyz,sign= pmov_map[mov]
          dp_curr[ixyz]+= sign*acceleration*dt
          if dp_curr[ixyz]>speed_max:  dp_curr[ixyz]= speed_max
          if dp_curr[ixyz]<-speed_max:  dp_curr[ixyz]= -speed_max

        for ixyz in range(3):
          if mov in list(pmov_map.keys()) and ixyz==pmov_map[mov][0]:  continue
          if dp_curr[ixyz]>acceleration*dt:  dp_curr[ixyz]-= acceleration*dt
          if dp_curr[ixyz]<-acceleration*dt:  dp_curr[ixyz]+= acceleration*dt
          if abs(dp_curr[ixyz])<=acceleration*dt:  dp_curr[ixyz]= 0.0

        print('dp_curr',dp_curr)
        dq= ( np.linalg.pinv(ur.J(ur.Q()))*(np.mat(dp_curr+[0,0,0]).T) ).ravel().tolist()[0]
        #dq= [0.0]*6
        #print 'dq',dq
        velctrl.Step(dq)

        rate.sleep()

  finally:
    is_running[0]= False
    t1.join()

