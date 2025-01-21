#!/usr/bin/python3
#Switching to enable and disable robot by pressing a key.
from dxl_cranex7 import *
import time
from kbhit2 import TKBHit

#Setup the device
crane= TCraneX7()
crane.Setup()
#crane.EnableTorque()
#crane.SetPWM({jname:0 for jname in crane.JointNames()})

state= 'disabled'

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      state= {'enabled':'disabled', 'disabled':'enabled'}[state]
      if state=='enabled':
        crane.EnableTorque()
      else:
        crane.DisableTorque()
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

#crane.DisableTorque()
crane.Quit()

