#!/usr/bin/python
#FactoryReset Dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()

print 'FactoryReset Dynamixel...'
dxl.FactoryReset()

#dxl.DisableTorque()
dxl.Quit()
