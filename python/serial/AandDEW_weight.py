#!/usr/bin/python
#\file    AandDEW_weight.py
#\brief   Read data from the A and D EW weight.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.26, 2019
import sys
import serial

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyUSB0'
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 2400

  ser= serial.Serial(dev,baudrate,serial.SEVENBITS,serial.PARITY_EVEN)

  try:
    while True:
      raw= ser.readline()
      if len(raw)!=17:
        value= None
        continue
      else:
        value= float(raw[3:12])
      print '"{raw}" / {v} ({l})'.format(raw=repr(raw), v=value, l=len(raw))

  finally:
    ser.close()
