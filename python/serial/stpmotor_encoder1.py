#!/usr/bin/python3
#\file    stpmotor_encoder1.py
#\brief   Stepping motor control with reading encoder value via Arduino.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.06, 2023
import sys
import serial
import threading
import time
from kbhit2 import TKBHit

class TSteppingMotorLinearEncoder(object):
  def __init__(self, dev='/dev/ttyACM0', br=2e6, start=True, disp=False):
    #serial.SEVENBITS
    self.ser= serial.Serial(dev,br,serial.EIGHTBITS,serial.PARITY_NONE)
    self.ser.reset_input_buffer()
    self.ser.reset_output_buffer()
    self.disp= disp
    self.locker= threading.RLock()
    if start: self.Start()

  def Start(self):
    self.running= True
    self.n= 0
    self.raw= None
    self.value= None
    self.th= threading.Thread(name='TSteppingMotorLinearEncoder', target=self.ReadingLoop)
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

  def ReadingLoop(self):
    try:
      n= 0
      while self.running:
        raw= self.ser.readline()
        try:
          value= float(raw.strip())
        except ValueError:
          value= None
          continue

        n+= 1

        with self.locker:
          self.n= n
          self.raw= raw
          self.value= value

        #if n%100==0:
        if self.disp: print('{n} Received: {raw} ({l}), {v}'.format(n=n, raw=repr(raw), l=len(raw), v=value))
        #time.sleep(0.005)

    finally:
      self.ser.close()

  def WriteSpeed(self, value):
    self.ser.write(chr(value))


if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyACM0'
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 115200

  ctrl= TSteppingMotorLinearEncoder(dev, baudrate, disp=False)
  try:
    with TKBHit() as kbhit:
      while True:
        if kbhit.IsActive():
          key= kbhit.KBHit()
          if key=='q':
            break;
          elif key=='z':
            ctrl.WriteSpeed(50)
          elif key=='x':
            ctrl.WriteSpeed(10)
          elif key=='c':
            ctrl.WriteSpeed(128+10)
          elif key=='v':
            ctrl.WriteSpeed(128+50)
        else:
          break

        if not isinstance(ctrl.Value,float):  print('LIN invalid value:',ctrl.Raw); time.sleep(0.1); continue
        print('{n} Latest: {raw} ({l}), {v}'.format(n=ctrl.N, raw=repr(ctrl.Raw), l=len(ctrl.Raw), v=ctrl.Value))
        time.sleep(0.001)

  except KeyboardInterrupt:
    pass

  finally:
    ctrl.Stop()
