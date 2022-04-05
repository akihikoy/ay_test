#!/usr/bin/python
#\file    dxl_fd2f4dof.py
#\brief   Control module of FD2F4DoF gripper.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.15, 2022
from dxl_mikata import *

class TFD2F4DoF(TMikata):
  def __init__(self, dev='/dev/ttyUSB0'):
    self.dev= dev
    self.baudrate= 2e6
    self.dxl_type= ['XM430-W350']*4
    self.dxl_ids= [1,2,3,4]
    self.joint_names= ['joint_1', 'joint_2', 'joint_3', 'joint_4']
    self.dxl= {}  #{joint_name:TDynamixel1}
    self.op_mode= 'POSITION'
    self.goal_pwm= [100]*4

    self.port_locker= threading.RLock()
    self.state_locker= threading.RLock()
    self.state= {'stamp':0.0, 'name':self.JointNames(), 'position':[], 'velocity':[], 'effort':[]}
    self.hz_state_obs= 50  #State observation rate (Hz).
    self.hz_traj_ctrl= 200  #Trajectory control rate (Hz).
    self.threads= {  #ThreadName:[IsActive,ThreadObject]
      'StateObserver':[False,None],
      'TrajectoryController':[False,None],}
