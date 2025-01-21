#!/usr/bin/python3
#Printing current current.

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.OpMode= 'CURRPOS'
dxl.Setup()
dxl.EnableTorque()

try:
  while True:
    print('Position: {0} \t Velocity: {1} \t Current: {2} \t PWM: {3}'.format(dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM()))
    #print 'Position,Current=',dxl.Position(),'{0:16b}'.format(dxl.Current())
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#dxl.DisableTorque()
dxl.Quit()
