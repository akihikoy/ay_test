#!/usr/bin/python3
#\file    dxl_mikata.py
#\brief   Control module of Mikata Arm.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.22, 2018
from dxl_util import *

import math
import time
import threading
import copy
from rate_adjust import TRate
from cubic_hermite_spline import TCubicHermiteSpline

class TMikata(object):
  def __init__(self, dev='/dev/ttyUSB0'):
    self.dev= dev
    self.baudrate= 1e6
    self.dxl_type= ['XM430-W350']*5
    self.dxl_ids= [1,2,3,4,5]
    self.joint_names= ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'gripper_joint_5']
    self.dxl= {}  #{joint_name:TDynamixel1}
    self.op_mode= 'POSITION'
    #self.goal_pwm= [10, 20, 15, 10, 12]
    self.goal_pwm= [70, 50, 40, 40, 30]
    #self.goal_pwm= [50,50,50,50,50]
    #self.goal_pwm= [90,90,90,90,90]

    self.port_locker= threading.RLock()
    self.state_locker= threading.RLock()
    self.state= {'stamp':0.0, 'name':self.JointNames(), 'position':[], 'velocity':[], 'effort':[]}
    self.hz_state_obs= 50  #State observation rate (Hz).
    self.hz_traj_ctrl= 200  #Trajectory control rate (Hz).
    self.threads= {  #ThreadName:[IsActive,ThreadObject]
      'StateObserver':[False,None],
      'TrajectoryController':[False,None],}

  def _joint_names(self,joint_names):
    return joint_names if joint_names is not None else self.joint_names

  def JointNames(self):
    return self.joint_names

  def Setup(self):
    with self.port_locker:
      self.dxl= {jname:TDynamixel1(type,self.dev) for type,jname in zip(self.dxl_type,self.joint_names)}
      for id,jname in zip(self.dxl_ids,self.joint_names):
        dxl= self.dxl[jname]
        dxl.OpMode= self.op_mode
        dxl.Id= id
        dxl.Baudrate= self.baudrate
        if not dxl.Setup():
          print('Failed to setup Dynamixel at:',jname)
          return False

    #Conversions from/to Dynamixel value to/from PWM(percentage), current(mA),
    #  velocity(rad/s), position(rad), temperature(deg of Celsius).
    self.conv_pwm= {jname:self.dxl[jname].ConvPWM for jname in self.joint_names}
    self.conv_curr= {jname:self.dxl[jname].ConvCurr for jname in self.joint_names}
    self.conv_vel= {jname:self.dxl[jname].ConvVel for jname in self.joint_names}
    self.conv_pos= {jname:self.dxl[jname].ConvPos for jname in self.joint_names}
    self.conv_temp= {jname:self.dxl[jname].ConvTemp for jname in self.joint_names}
    self.invconv_pwm= {jname:self.dxl[jname].InvConvPWM for jname in self.joint_names}
    self.invconv_curr= {jname:self.dxl[jname].InvConvCurr for jname in self.joint_names}
    self.invconv_vel= {jname:self.dxl[jname].InvConvVel for jname in self.joint_names}
    self.invconv_pos= {jname:self.dxl[jname].InvConvPos for jname in self.joint_names}
    self.invconv_temp= {jname:self.dxl[jname].InvConvTemp for jname in self.joint_names}

    self.SetPWM({jname:goal_pwm for jname,goal_pwm in zip(self.joint_names,self.goal_pwm)})
    return True

  def Quit(self):
    self.StopTrajectory()
    self.StopStateObs()
    for dxl in list(self.dxl.values()):
      dxl.Quit()

  def EnableTorque(self,joint_names=None):
    with self.port_locker:
      for jname in self._joint_names(joint_names):
        self.dxl[jname].EnableTorque()

  def DisableTorque(self,joint_names=None):
    with self.port_locker:
      for jname in self._joint_names(joint_names):
        self.dxl[jname].DisableTorque()

  def Reboot(self,joint_names=None):
    with self.port_locker:
      for jname in self._joint_names(joint_names):
        self.dxl[jname].Reboot()

  #Get current PWM.
  #  joint_names: Names of observing joints.
  #  as_dict: If True, the result is returned as a dictionary {joint_name:value}.
  #           Otherwise(default), the result is a list of values corresponding with joint_names.
  def PWM(self,joint_names=None,as_dict=False):
    joint_names= self._joint_names(joint_names)
    with self.port_locker:
      values= [self.dxl[jname].PWM() for jname in joint_names]
    values= [self.conv_pwm[jname](value) for (jname,value) in zip(joint_names,values)]
    if as_dict:  return {jname:value for (jname,value) in zip(joint_names,values)}
    else:        return values

  #Get current current.
  #  joint_names: Names of observing joints.
  #  as_dict: If True, the result is returned as a dictionary {joint_name:value}.
  #           Otherwise(default), the result is a list of values corresponding with joint_names.
  def Current(self,joint_names=None,as_dict=False):
    joint_names= self._joint_names(joint_names)
    with self.port_locker:
      values= [self.dxl[jname].Current() for jname in joint_names]
    values= [self.conv_curr[jname](value) for (jname,value) in zip(joint_names,values)]
    if as_dict:  return {jname:value for (jname,value) in zip(joint_names,values)}
    else:        return values

  #Get current velocity.
  #  joint_names: Names of observing joints.
  #  as_dict: If True, the result is returned as a dictionary {joint_name:value}.
  #           Otherwise(default), the result is a list of values corresponding with joint_names.
  def Velocity(self,joint_names=None,as_dict=False):
    joint_names= self._joint_names(joint_names)
    with self.port_locker:
      values= [self.dxl[jname].Velocity() for jname in joint_names]
    values= [self.conv_vel[jname](value) for (jname,value) in zip(joint_names,values)]
    if as_dict:  return {jname:value for (jname,value) in zip(joint_names,values)}
    else:        return values

  #Get current position.
  #  joint_names: Names of observing joints.
  #  as_dict: If True, the result is returned as a dictionary {joint_name:value}.
  #           Otherwise(default), the result is a list of values corresponding with joint_names.
  def Position(self,joint_names=None,as_dict=False):
    joint_names= self._joint_names(joint_names)
    with self.port_locker:
      values= [self.dxl[jname].Position() for jname in joint_names]
    values= [self.conv_pos[jname](value) for (jname,value) in zip(joint_names,values)]
    if as_dict:  return {jname:value for (jname,value) in zip(joint_names,values)}
    else:        return values

  #Get current temperature.
  #  joint_names: Names of observing joints.
  #  as_dict: If True, the result is returned as a dictionary {joint_name:value}.
  #           Otherwise(default), the result is a list of values corresponding with joint_names.
  def Temperature(self,joint_names=None,as_dict=False):
    joint_names= self._joint_names(joint_names)
    with self.port_locker:
      values= [self.dxl[jname].Temperature() for jname in joint_names]
    values= [self.conv_temp[jname](value) for (jname,value) in zip(joint_names,values)]
    if as_dict:  return {jname:value for (jname,value) in zip(joint_names,values)}
    else:        return values

  #Move the position to a given value(rad).
  #  target: Target positions {joint_name:position(rad)}
  #  blocking: True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveTo(self, target, blocking=True, threshold=0.05):
    with self.port_locker:
      for jname,pos in target.items():
        self.dxl[jname].MoveTo(self.invconv_pos[jname](pos),blocking=False)

    while blocking:
      pos= self.Position(list(target.keys()))
      if None in pos:  return
      #print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.Id, target, pos))
      if all((abs(trg-p)<=threshold for trg,p in zip(list(target.values()),pos))):  break

  #Move the position to a given value(rad) with given current(mA).
  #  target: Target positions and currents {joint_name:(position(rad),current(mA))}
  #  blocking: True: this function waits the target position is reached.  False: this function returns immediately.
  def MoveToC(self, target, blocking=True, threshold=0.05):
    with self.port_locker:
      for jname,(pos,curr) in target.items():
        self.dxl[jname].MoveToC(self.invconv_pos[jname](pos),self.invconv_curr[jname](curr),blocking=False)

    while blocking:
      pos= self.Position(list(target.keys()))
      if None in pos:  return
      #print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.Id, target, pos))
      if all((abs(trg[0]-p)<=threshold for trg,p in zip(list(target.values()),pos))):  break

  #Set current(mA)
  #  current: Target currents {joint_name:current(mA)}
  def SetCurrent(self, current):
    with self.port_locker:
      for jname,curr in current.items():
        self.dxl[jname].SetCurrent(self.invconv_curr[jname](curr))

  #Set velocity(rad/s)
  #  velocity: Target velocities {joint_name:velocity(rad/s)}
  def SetVelocity(self, velocity):
    with self.port_locker:
      for jname,vel in velocity.items():
        self.dxl[jname].SetVelocity(self.invconv_vel[jname](vel))

  #Set PWM(percentage).
  #  pwm: Target PWMs {joint_name:pwm(percentage)}
  def SetPWM(self, pwm):
    with self.port_locker:
      for jname,pwm_ in pwm.items():
        self.dxl[jname].SetPWM(self.invconv_pwm[jname](pwm_))

  #Get current state saved in memory (no port access when running this function).
  #Run StartStateObs before using this.
  def State(self):
    with self.state_locker:
      state= copy.deepcopy(self.state)
    return state

  #Start state observation.
  #  callback: Callback function at the end of each observation cycle.
  #           callback may return True or False. If False is returned, the thread stops.
  def StartStateObs(self, callback=None):
    self.StopStateObs()
    th_func= lambda:self.StateObserver(callback)
    self._state_observer_callback= callback  #For future use.
    self.threads['StateObserver']= [True, threading.Thread(name='StateObserver', target=th_func)]
    self.threads['StateObserver'][1].start()

  #Stop state observation.
  def StopStateObs(self):
    if self.threads['StateObserver'][0]:
      self.threads['StateObserver'][0]= False
      self.threads['StateObserver'][1].join()
    self.threads['StateObserver']= [False,None]

  #Set rate (Hz) of state observation.
  #Works anytime.
  def SetStateObsRate(self, rate):
    if self.hz_state_obs!=rate:
      self.hz_state_obs= rate
      if self.threads['StateObserver'][0]:
        self.StartStateObs(self._state_observer_callback)

  #Follow a trajectory.
  #  joint_names: Names of joints to be controlled.
  #  (q_traj,t_traj): Sequence of (joint positions) and (time from start).
  #  current: Currents of joints (available when the operation mode=CURRPOS).
  #  callback: Callback called in the control loop.  If not None, it is used in the cases:
  #      1. At the beginning of loop: callback('loop_begin',t,q,dq)
  #         (t: elapsed time, q: target joint positions, dq: joint velocities).
  #         callback may return True (continue) or False (stop).
  #      2. At the end of loop: callback('loop_end',None,None,None).
  #      3. At the end of control: callback('final',None,None,None).
  def FollowTrajectory(self, joint_names, q_traj, t_traj, current=None, blocking=False, callback=None):
    self.StopTrajectory()
    th_func= lambda:self.TrajectoryController(joint_names, q_traj, t_traj, current, callback)
    self.threads['TrajectoryController']= [True, threading.Thread(name='TrajectoryController', target=th_func)]
    self.threads['TrajectoryController'][1].start()
    if blocking:
      self.threads['TrajectoryController'][1].join()
      self.StopTrajectory()

  #Stop following the trajectory.
  def StopTrajectory(self):
    if self.threads['TrajectoryController'][0]:
      self.threads['TrajectoryController'][0]= False
      self.threads['TrajectoryController'][1].join()
    self.threads['TrajectoryController']= [False,None]

  #Set rate (Hz) of trajectory following controller.
  def SetTrajectoryCtrlRate(self, rate):
    if self.hz_traj_ctrl!=rate:
      self.hz_traj_ctrl= rate

  #State observer thread.
  #NOTE: Don't call this function directly.  Use self.StartStateObs and self.State
  def StateObserver(self, callback):
    rate= TRate(self.hz_state_obs)
    while self.threads['StateObserver'][0]:
      with self.port_locker:
        state= {
          'stamp':time.time(),
          'name':self.JointNames(),
          'position':self.Position(self.JointNames()),
          'velocity':self.Velocity(self.JointNames()),
          'effort':self.PWM(self.JointNames()),  #FIXME: PWM vs. Current
          }
      with self.state_locker:
        self.state= state
      if callback is not None:
        if callback(state)==False:  break
      #print state['position']
      rate.sleep()
    self.threads['StateObserver'][0]= False

  #Trajectory controller thread.
  #NOTE: Don't call this function directly.  Use self.FollowTrajectory
  def TrajectoryController(self, joint_names, q_traj, t_traj, current, callback):
    assert(len(t_traj)>0)
    assert(len(q_traj)==len(t_traj))
    assert(len(joint_names)==len(q_traj[0]))
    dof= len(joint_names)

    #Revising trajectory (add an initial point).
    if t_traj[0]>1.0e-3:
      q_traj= [self.Position(joint_names)]+q_traj
      t_traj= [0.0]+t_traj

    #Modeling the trajectory with spline.
    splines= [TCubicHermiteSpline() for d in range(dof)]
    for d in range(len(splines)):
      data_d= [[t,q[d]] for q,t in zip(q_traj,t_traj)]
      splines[d].Initialize(data_d, tan_method=splines[d].CARDINAL, c=0.0, m=0.0)

    rate= TRate(self.hz_traj_ctrl)
    t0= time.time()
    while all(((time.time()-t0)<t_traj[-1], self.threads['TrajectoryController'][0])):
      t= time.time()-t0
      #q= [splines[d].Evaluate(t) for d in xrange(dof)]
      q_dq= [splines[d].Evaluate(t,with_tan=True) for d in range(dof)]
      q= [q for q,_ in q_dq]
      dq= [dq for _,dq in q_dq]
      if callback is not None:
        if callback('loop_begin',t,q,dq)==False:  break
      #print t, q
      if current is None:
        with self.port_locker:
          self.MoveTo({jname:qj for jname,qj in zip(joint_names,q)}, blocking=False)
      else:
        with self.port_locker:
          self.MoveToC({jname:(qj,ej) for jname,qj,ej in zip(joint_names,q,current)}, blocking=False)
      if callback is not None:
        callback('loop_end',None,None,None)
      rate.sleep()

    self.threads['TrajectoryController'][0]= False
    if callback is not None:
      callback('final',None,None,None)
