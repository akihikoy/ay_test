#!/usr/bin/python3
#Record key points by moving the arm.

from dxl_fd2f4dof import *
import time
from kbhit2 import TKBHit

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()

#Relax mode:
gripper.EnableTorque()
gripper.SetPWM({jname:0 for jname in gripper.JointNames()})
#Disabled mode:
#gripper.DisableTorque()

gripper.StartStateObs()

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      #print gripper.State()
      #print gripper.State()['position']
      print('[',', '.join(['{:.4f}'.format(p) for p in gripper.Position()]),'],')
    #gripper.MoveTo(gripper.Position(as_dict=True),blocking=False)
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

gripper.StopStateObs()

#gripper.DisableTorque()
gripper.Quit()
