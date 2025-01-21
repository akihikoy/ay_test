#!/usr/bin/python3
#\file    loadcell1.py
#\brief   Serial communication test with Arduino where loadcells are installed.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.13, 2021

import sys
import serial
import time

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyACM0'
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 2e6

  ser= serial.Serial(dev,baudrate,serial.SEVENBITS,serial.PARITY_NONE)
  ser.reset_input_buffer()
  ser.reset_output_buffer()

  count= 0
  t_prev= time.time()
  try:
    while True:
      raw= ser.readline()
      try:
        value= float(raw.replace('Reading:','').replace('g\r\n','').strip())
      except ValueError:
        print('No regular value: {raw} ({l})'.format(raw=repr(raw), l=len(raw)))
        continue
      print('{raw} ({l}), {value}'.format(raw=repr(raw), l=len(raw), value=value))
      #if len(raw)!=17:  continue
      #value= float(raw[3:12])
      count+= 1
      if count%40==0:
        print('FPS:',40./(time.time()-t_prev))
        t_prev= time.time()

      #ser.reset_input_buffer()
      ##ser.flushInput()
      #raw= ser.readline()
      #time.sleep(0.01)

  finally:
    ser.close()

