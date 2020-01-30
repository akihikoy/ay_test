#!/usr/bin/python
#Change baud rate.

from dxl_util import *
from _config import *

#Setup the device
dxl= [TDynamixel1(DXL_TYPE) for _ in DXL_ID]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()

for i,_ in enumerate(DXL_ID):
  dxl[i].Write('BAUD_RATE',dxl[i].BAUD_RATE.index(2e6))

for i,_ in enumerate(DXL_ID):
  dxl[i].Quit()
