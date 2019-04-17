#!/usr/bin/python
#Printing current position without enabling torque.

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
#dxl.EnableTorque()

try:
  while True:
    print 'Position=',dxl.Position()
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

#dxl.DisableTorque()
dxl.Quit()
