#!/usr/bin/python
#Test of RHP12RN (Thormang gripper).

from dxl_util import *
from _config import *
from rate_adjust import TRate
import time,sys,math
import numpy as np

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

stdout= sys.stdout
with open('/tmp/test_status.dat','w') as fp:
  sys.stdout= fp
  print 'Status:'
  dxl.PrintStatus()
  print '-----'
  print 'Hardware error status:'
  dxl.PrintHardwareErrSt()
  print '-----'
  print 'Shutdown configuration:'
  dxl.PrintShutdown()
  print '-----'
sys.stdout= stdout

with open('/tmp/test_motion.dat','w') as fp:
  sys.stdout= fp
  #Move to initial position
  p_start= 100
  dxl.MoveTo(p_start)
  time.sleep(0.5)
  print 0.0, p_start, dxl.Position(), dxl.Velocity()

  t_start= time.time()
  rate= TRate(100)
  #Move to a target position
  for t in np.mgrid[0:4*math.pi:0.01]:
    p_trg= p_start + 500*(0.5-0.5*math.cos(t))
    dxl.MoveTo(p_trg, blocking=False)
    rate.sleep()
    print time.time()-t_start, p_trg, dxl.Position(), dxl.Velocity()
sys.stdout= stdout

dxl.DisableTorque()
dxl.Quit()
