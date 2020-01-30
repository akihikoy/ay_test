#!/usr/bin/python
#Enable dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= [TDynamixel1(DXL_TYPE) for _ in DXL_ID]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()
  dxl[i].EnableTorque()

#for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
for i,_ in enumerate(DXL_ID):
  dxl[i].Quit()
