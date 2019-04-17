#!/usr/bin/python
#Print dynamixel status.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()

print 'Status:'
dxl.PrintStatus()
print 'Hardware error status:'
dxl.PrintHardwareErrSt()
print 'Shutdown configuration:'
dxl.PrintShutdown()

#dxl.DisableTorque()
dxl.Quit()
