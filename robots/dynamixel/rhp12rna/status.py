#!/usr/bin/python
#Print dynamixel status.

from dxl_util import *
from _config import *

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()

print '-----'
print 'Status:'
dxl.PrintStatus()
print '-----'
print 'Hardware error status:'
dxl.PrintHardwareErrSt()
print '-----'
print 'Shutdown configuration:'
dxl.PrintShutdown()
print '-----'

#dxl.DisableTorque()
dxl.Quit()
