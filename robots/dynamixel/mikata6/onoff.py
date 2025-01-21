#!/usr/bin/python3
#Switching to enable and disable robot by pressing a key.
from dxl_mikata6 import *
import time
from kbhit2 import TKBHit

#Setup the device
mikata= TMikata6()
mikata.Setup()
#mikata.EnableTorque()
#mikata.SetPWM({jname:0 for jname in mikata.JointNames()})

state= 'disabled'

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      state= {'enabled':'disabled', 'disabled':'enabled'}[state]
      if state=='enabled':
        mikata.EnableTorque()
      else:
        mikata.DisableTorque()
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

#mikata.DisableTorque()
mikata.Quit()

