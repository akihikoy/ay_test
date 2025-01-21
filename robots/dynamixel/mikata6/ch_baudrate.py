#!/usr/bin/python3
#Change baud rate.

from dxl_mikata6 import *

BAUD_RATE_FROM= 1e6
BAUD_RATE_TO= 3e6

mikata= TMikata6()
mikata.baudrate= BAUD_RATE_FROM
mikata.Setup()

for i,jname in enumerate(mikata.JointNames()):
  dxl= mikata.dxl[jname]
  print('Changing baud rate of joint {jname} (id:{id})'.format(jname=jname,id=mikata.dxl_ids[i]))
  dxl.Write('BAUD_RATE',dxl.BAUD_RATE.index(BAUD_RATE_TO))

mikata.Quit()
