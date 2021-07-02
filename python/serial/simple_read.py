#!/usr/bin/python
#\file    simple_read.py
#\brief   Simply read from serial port.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.01, 2021

import sys
import serial

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyACM0'
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 19200

  #serial.SEVENBITS
  ser= serial.Serial(dev,baudrate,serial.EIGHTBITS,serial.PARITY_NONE)

  try:
    while True:
      raw= ser.readline()
      print 'Received: {raw} ({l})'.format(raw=repr(raw), l=len(raw))

  finally:
    ser.close()

