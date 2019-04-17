#!/usr/bin/python
#Disable dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Setup()
#dxl.EnableTorque()

dxl.DisableTorque()
dxl.Quit()
