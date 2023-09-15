#!/usr/bin/python
#\file    iolink_geh6060il_1.py
#\brief   Test code of operating the GEH6060IL gripper through the IO-Link USB master.
#         NOTE: It turned out that this does not work.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.14, 2023
from __future__ import print_function
import sys
import serial
import time
import struct

PROCESS_DATA_FORMAT= [
    ('Control_Word', 2),
    ('Device_Mode', 1),
    ('Workpiece_No', 1),
    ('Reserve', 1),
    ('Position_Tolerance', 1),
    ('Grip_Force', 1),
    ('Drive_Velocity', 1),
    ('Base_Position', 2),
    ('Shift_Position', 2),
    ('Teach_Position', 2),
    ('Work_Position', 2),
  ]
PROCESS_DATA_FORMAT_STR= ''.join([{1:'B',2:'H'}[size] for key,size in PROCESS_DATA_FORMAT])
print('PROCESS_DATA_FORMAT_STR= {}'.format(PROCESS_DATA_FORMAT_STR))

def PackProcessData(data):
  seq= [data[key] for key,size in PROCESS_DATA_FORMAT]
  return struct.pack(PROCESS_DATA_FORMAT_STR, *seq)

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyS31'
  baudrate= 38400
  ser= serial.Serial(dev,baudrate,serial.EIGHTBITS,serial.PARITY_NONE)

  data= {key:0 for key,size in PROCESS_DATA_FORMAT}

  try:
    packet= PackProcessData(data)

    #Send 16 byte data.
    ser.write(packet)

    #Receive 6 byte data.
    raw= ser.readline()
    print('Received: {} ({} / 6)'.format(raw=repr(raw), l=len(raw)))

  finally:
    ser.close()

