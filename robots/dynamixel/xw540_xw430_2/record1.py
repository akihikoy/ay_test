#!/usr/bin/python3
#Record key points by moving the arm.

from dxl_xw540_xw430 import *
import time
from kbhit2 import TKBHit

#Setup the device
device= TXW540XW430()
device.Setup()

#Relax mode:
#device.EnableTorque()
#device.SetPWM({jname:0 for jname in device.JointNames()})
#Disabled mode:
device.DisableTorque()

device.StartStateObs()

try:
  kbhit= TKBHit()
  while True:
    c= kbhit.KBHit()
    if c=='q':  break
    elif c is not None:
      #print device.State()
      #print device.State()['position']
      print('[',', '.join(['{:.4f}'.format(p) for p in device.Position()]),'],')
    #device.MoveTo(device.Position(as_dict=True),blocking=False)
    time.sleep(0.0025)
except KeyboardInterrupt:
  pass

device.StopStateObs()

#device.DisableTorque()
device.Quit()
