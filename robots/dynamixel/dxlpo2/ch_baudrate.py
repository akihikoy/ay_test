#!/usr/bin/python
#Change baud rate.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.CurrentLimit= CURRENT_LIMIT
dxl.Setup()

dxl.Write('BAUD_RATE',dxl.BAUD_RATE.index(2e6))

dxl.Quit()
