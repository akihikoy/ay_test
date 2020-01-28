#!/usr/bin/python
#Printing current position without enabling torque.

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.CurrentLimit= CURRENT_LIMIT
dxl.Setup()
#dxl.EnableTorque()

try:
  i= 0
  t0= time.time()
  while True:
    print 'Position=',dxl.Position()
    i+= 1
    #time.sleep(0.001)
except KeyboardInterrupt:
  print 'Observation freq:',i/(time.time()-t0),'Hz'

#dxl.DisableTorque()
dxl.Quit()
