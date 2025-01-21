#!/usr/bin/python3
#Change baud rate.

from dxl_util import *
from _config import *

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i]) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  dxl[i].CurrentLimit= CURRENT_LIMIT
  dxl[i].Setup()

for i,_ in enumerate(DXL_ID):
  dxl[i].Write('BAUD_RATE',dxl[i].BAUD_RATE.index(2e6))

for i,_ in enumerate(DXL_ID):
  dxl[i].Quit()
