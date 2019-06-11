#!/usr/bin/python
#Enable dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#dxl.DisableTorque()
dxl.Quit()
