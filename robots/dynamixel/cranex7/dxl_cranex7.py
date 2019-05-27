#!/usr/bin/python
#\file    dxl_cranex7.py
#\brief   Control module of Crane-X7 Arm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.01, 2018
from dxl_mikata import *

class TCraneX7(TMikata):
  def __init__(self, dev='/dev/ttyUSB0'):
    self.dev= dev
    self.baudrate= 3e6
    self.dxl_type= ['XM430-W350']*8  # Actually XM540-W270-R + XM430-W350-R
    self.dxl_ids= [2,3,4,5,6,7,8,9]
    self.joint_names= ['crane_x7_shoulder_fixed_part_pan_joint',
        'crane_x7_shoulder_revolute_part_tilt_joint',
        'crane_x7_upper_arm_revolute_part_twist_joint',
        'crane_x7_upper_arm_revolute_part_rotate_joint',
        'crane_x7_lower_arm_fixed_part_joint',
        'crane_x7_lower_arm_revolute_part_joint',
        'crane_x7_wrist_joint',
        'crane_x7_gripper_finger_a_joint']
    self.dxl= {}  #{joint_name:TDynamixel1}
    self.op_mode= 'POSITION'
    self.goal_pwm= [80]*8

    self.port_locker= threading.RLock()
    self.state_locker= threading.RLock()
    self.state= {'stamp':0.0, 'name':self.JointNames(), 'position':[], 'velocity':[], 'effort':[]}
    self.hz_state_obs= 50  #State observation rate (Hz).
    self.hz_traj_ctrl= 200  #Trajectory control rate (Hz).
    self.threads= {  #ThreadName:[IsActive,ThreadObject]
      'StateObserver':[False,None],
      'TrajectoryController':[False,None],}
