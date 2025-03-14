#!/usr/bin/python3
#\file    AandDEW_logger.py
#\brief   Weight logger with A and D digital weight.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.16, 2022
import sys
import serial
from kbhit2 import TKBHit
from time_str import TimeStr

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyUSB0'  #ttyS3:Com3(win); ttyUSB0
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 2400

  ser= serial.Serial(dev,baudrate,serial.SEVENBITS,serial.PARITY_EVEN)
  ser.reset_input_buffer()
  ser.reset_output_buffer()

  log_fp= None
  log_file_name= 'weight_log_{}.dat'.format(TimeStr('normal'))

  value_prev= None

  try:
    with TKBHit() as kbhit:
      while True:
        request= None
        if kbhit.IsActive():
          key= kbhit.KBHit()
          if key=='q':
            break;
          elif key in ('p',):
            request= 'print'
          elif key in (' ','l'):
            request= 'log'
        else:
          break
        raw= ser.readline()
        if len(raw)!=17:
          value= None
          print('Invalid data')
          continue
        else:
          value= float(raw[3:12])
        if request=='print':
          print('"{raw}" / {v} ({l})'.format(raw=repr(raw), v=value, l=len(raw)))
        elif request=='log':
          if log_fp is None:
            log_fp= open(log_file_name,'w')
            print('Opened a log file:',log_file_name)
          log_fp.write('{}\n'.format(value))
          print('Recorded: {} ({})'.format(value, None if value_prev is None else value-value_prev))
          value_prev= value
  finally:
    if log_fp is not None:  log_fp.close()
    ser.close()

