#!/usr/bin/python
#\file    dxl_mikata6.py
#\brief   Control module of 6DoF Mikata Arm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.31, 2019
from dxl_mikata import *

class TMikata6(TMikata):
  def __init__(self, dev='/dev/ttyUSB0'):
    self.dev= dev
    self.baudrate= 3e6
    self.dxl_type= ['XM430-W350']*8  # Actually ... + ...
    self.dxl_ids= [11,12,13,14,15,16,17]
    self.joint_names= ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    self.dxl= {}  #{joint_name:TDynamixel1}
    self.op_mode= 'POSITION'
    self.goal_pwm= [80]*7

    self.port_locker= threading.RLock()
    self.state_locker= threading.RLock()
    self.state= {'stamp':0.0, 'name':self.JointNames(), 'position':[], 'velocity':[], 'effort':[]}
    self.hz_state_obs= 50  #State observation rate (Hz).
    self.hz_traj_ctrl= 200  #Trajectory control rate (Hz).
    self.threads= {  #ThreadName:[IsActive,ThreadObject]
      'StateObserver':[False,None],
      'TrajectoryController':[False,None],}
