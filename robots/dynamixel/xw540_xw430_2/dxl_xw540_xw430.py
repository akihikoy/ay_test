#!/usr/bin/python3
#\file    dxl_xw540_xw430.py
#\brief   Control module of XW540 and XW430 composite.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.21, 2025
from dxl_mikata import *

class TXW540XW430(TMikata):
  def __init__(self, dev='/dev/ttyUSB0'):
    self.dev= dev
    self.baudrate= 2e6
    self.dxl_type= ['XW540-T260', 'XW430-T333']
    self.dxl_ids= [2,3]
    self.joint_names= ['joint_push', 'joint_cut']
    self.dxl= {}  #{joint_name:TDynamixel1}
    self.op_mode= 'POSITION'
    self.goal_pwm= [100]*2

    self.port_locker= threading.RLock()
    self.state_locker= threading.RLock()
    self.state= {'stamp':0.0, 'name':self.JointNames(), 'position':[], 'velocity':[], 'effort':[]}
    self.hz_state_obs= 50  #State observation rate (Hz).
    self.hz_traj_ctrl= 50  #Trajectory control rate (Hz).
    self.threads= {  #ThreadName:[IsActive,ThreadObject]
      'StateObserver':[False,None],
      'TrajectoryController':[False,None],}
