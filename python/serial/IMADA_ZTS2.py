#!/usr/bin/python3
#\file    IMADA_ZTS2.py
#\brief   Loading data from IMADA ZTS force sensor (class version running on a thread);
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.19, 2021
from IMADA_ZTS import ZTS_UNIT_CODE
import sys
import serial
import threading
import time

class TZTS(object):
  def __init__(self, dev='/dev/ttyS3', br=19200, start=True, disp=False, cnt=False):
    #serial.SEVENBITS
    self.ser= serial.Serial(dev,br,serial.EIGHTBITS,serial.PARITY_NONE,serial.STOPBITS_ONE)
    self.ser.reset_input_buffer()
    self.ser.reset_output_buffer()
    self.disp= disp
    self.cnt= cnt  #Continuous mode.
    self.locker= threading.RLock()
    if start: self.Start()

  def Start(self):
    self.running= True
    self.n= 0
    self.raw= None
    self.value= None
    self.unit= None
    self.th= threading.Thread(name='TZTS', target=self.ReadingLoop)
    self.th.start()

  def Stop(self):
    self.running= False
    self.th.join()
    self.th= None

  @property
  def N(self):
    with self.locker:
      return self.n

  @property
  def Raw(self):
    with self.locker:
      return self.raw

  @property
  def Value(self):
    with self.locker:
      return self.value

  @property
  def Unit(self):
    with self.locker:
      return self.unit

  def ReadingLoop(self):
    '''
    XAR: Send a current value.
    XAg: Send data continuously at 10Hz.
    XAG: Send data continuously at 2000Hz.
    XAS: Stop sending data.
    XFC: Request a list of units.
    '''
    try:
      self.ser.write(b'XFC\r')
      raw= self.ser.read_until(b'\r').decode('utf-8')
      if raw[:3]=='XFC':
        units= [ZTS_UNIT_CODE[raw[i:i+2]] for i in range(3,15,2)]
      else:  units= ['None']*6
      print('Units:',units)

      if self.cnt:  self.ser.write(b'XAG\r')  #NOTE: Stop by XAS. No need to use sleep.
      n= 0
      while self.running:
        if not self.cnt:  self.ser.write(b'XAR\r')  #Alternative to XAR.
        raw= self.ser.read_until(b'\r').decode('utf-8')

        try:
          value= float(raw[1:7])
          unit= units[int(raw[15])]
        except ValueError:
          value= None
          unit= None
          continue

        n+= 1

        with self.locker:
          self.n= n
          self.raw= raw
          self.value= value
          self.unit= unit

        #if n%100==0:
        if self.disp: print('{n} Received: {raw} ({l}), {v} {u}'.format(n=n, raw=repr(raw), l=len(raw), v=value, u=unit))
        if not self.cnt:  time.sleep(0.005)
        #NOTE: If you want to set the sleep time zero (no sleep),
        #  it would be better to use the continuous data mode at 2000Hz,
        #  started by XAG and stopped by XAS. In this case, do not send XAR.

      if self.cnt:  self.ser.write(b'XAS\r')  #Use with XAG.

    finally:
      self.ser.close()


if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyS3'  #COM3(Win); /dev/ttyACM0
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 19200

  zts= TZTS(dev, baudrate)
  try:
    while True:
      time.sleep(0.1)
      print('{n} Latest: {raw} ({l}), {v} {u}'.format(n=zts.N, raw=repr(zts.Raw), l=len(zts.Raw), v=zts.Value, u=zts.Unit))

  except KeyboardInterrupt:
    pass

  finally:
    zts.Stop()
