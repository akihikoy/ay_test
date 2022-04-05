#!/usr/bin/python
#Switching to enable and disable robot by pressing a key.
from dxl_fd2f4dof import *
import time
from kbhit2 import TKBHit

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
#gripper.EnableTorque()
#gripper.SetPWM({jname:0 for jname in gripper.JointNames()})

state= 'disabled'

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      state= {'enabled':'disabled', 'disabled':'enabled'}[state]
      if state=='enabled':
        gripper.EnableTorque()
      else:
        gripper.DisableTorque()
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

#gripper.DisableTorque()
gripper.Quit()

