#!/usr/bin/python
#Reboot Dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()

print 'Rebooting Dynamixel...'
dxl.Reboot()

#dxl.DisableTorque()
dxl.Quit()
