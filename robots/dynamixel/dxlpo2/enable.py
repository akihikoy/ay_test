#!/usr/bin/python3
#Enable dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.CurrentLimit= CURRENT_LIMIT
dxl.Setup()
dxl.EnableTorque()

#dxl.DisableTorque()
dxl.Quit()
