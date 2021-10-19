#!/usr/bin/python
#\file    IMADA_ZTS.py
#\brief   Loading data from IMADA ZTS force sensor;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.07, 2021

import sys
import serial
import time

ZTS_UNIT_CODE= {
  '00':'None',
  '01':'mN',
  '02':'N',
  '03':'kN',
  '04':'g',
  '05':'kg',
  '07':'gf',
  '08':'kgf',
  '10':'ozf',
  '11':'lbf',
  '12':'klbf',
  '13':'N-cm',
  '14':'N-m',
  '16':'kgf-cm',
  '17':'kgf-m',
  '22':'ozf-in',
  '23':'lbf-in',
  }

if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyS3'  #COM3(Win); /dev/ttyACM0
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 19200

  #serial.SEVENBITS
  ser= serial.Serial(dev,baudrate,serial.EIGHTBITS,serial.PARITY_NONE,serial.STOPBITS_ONE)

  try:
    '''
    XAR: Send a current value.
    XAg: Send data continuously at 10Hz.
    XAG: Send data continuously at 2000Hz.
    XAS: Stop sending data.
    XFC: Request a list of units.
    '''
    ser.write('XFC\r')
    raw= ser.read_until('\r')
    if raw[:3]=='XFC':
      units= [ZTS_UNIT_CODE[raw[i:i+2]] for i in range(3,15,2)]
    else:  units= ['None']*6
    print 'Units:',units

    #ser.write('XAG\r')
    n= 0
    while True:
      ser.write('XAR\r')
      raw= ser.read_until('\r')

      try:
        value= float(raw[1:7])
        unit= units[int(raw[15])]
      except ValueError:
        value= None
        unit= None
        continue

      n+= 1
      #if n%100==0:
      print '{n} Received: {raw} ({l}), {v} {u}'.format(n=n, raw=repr(raw), l=len(raw), v=value, u=unit)
      time.sleep(0.01)
      #NOTE: If you want to set the sleep time zero (no sleep),
      #  it would be better to use the continuous data mode at 2000Hz,
      #  started by XAG and stopped by XAS. In this case, do not send XAR.

  except KeyboardInterrupt:
    #ser.write('XAS\r')
    pass

  finally:
    ser.close()

