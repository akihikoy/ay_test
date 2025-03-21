#!/usr/bin/python3
#Switching to enable and disable robot by pressing a key.
from dxl_xw540_xw430 import *
import time
from kbhit2 import TKBHit

#Setup the device
device= TXW540XW430()
device.Setup()
#device.EnableTorque()
#device.SetPWM({jname:0 for jname in device.JointNames()})

state= 'disabled'

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      state= {'enabled':'disabled', 'disabled':'enabled'}[state]
      if state=='enabled':
        device.EnableTorque()
      else:
        device.DisableTorque()
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

#device.DisableTorque()
device.Quit()

