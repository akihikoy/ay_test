#!/usr/bin/python
#Printing current positions of two Dynamixels without enabling torque.

from dxl_util import *
from _config import *
import time

#Setup the device
DXL_ID= [4,5]
BAUDRATE= 1e6
DXL_TYPE= 'XM430-W350'
dxl= [TDynamixel1(DXL_TYPE), TDynamixel1(DXL_TYPE)]
for i in (0,1):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()
  #dxl[i].EnableTorque()

try:
  while True:
    print 'Positions=',dxl[0].Position(),dxl[1].Position()
    time.sleep(0.001)
except KeyboardInterrupt:
  pass

for i in (0,1):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
