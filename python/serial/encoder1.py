#!/usr/bin/python
#\file    encoder1.py
#\brief   Loading data from a linear encoder.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.19, 2021
import sys
import serial
import threading
import time

class TLinearEncoder(object):
  def __init__(self, dev='/dev/ttyACM0', br=2e6, start=True, disp=False):
    #serial.SEVENBITS
    self.ser= serial.Serial(dev,br,serial.SEVENBITS,serial.PARITY_NONE)
    self.disp= disp
    self.locker= threading.RLock()
    if start: self.Start()

  def Start(self):
    self.running= True
    self.n= 0
    self.raw= None
    self.value= None
    self.th= threading.Thread(name='TLinearEncoder', target=self.ReadingLoop)
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
        if self.disp: print '{n} Received: {raw} ({l}), {v}'.format(n=n, raw=repr(raw), l=len(raw), v=value)
        #time.sleep(0.005)

    finally:
      self.ser.close()


if __name__=='__main__':
  dev= sys.argv[1] if len(sys.argv)>1 else '/dev/ttyACM0'
  baudrate= int(sys.argv[2]) if len(sys.argv)>2 else 2e6

  lin= TLinearEncoder(dev, baudrate, disp=False)
  try:
    while True:
      time.sleep(0.1)
      if not isinstance(lin.Value,float):  print 'LIN invalid value:',lin.Raw; time.sleep(0.1); continue
      print '{n} Latest: {raw} ({l}), {v}'.format(n=lin.N, raw=repr(lin.Raw), l=len(lin.Raw), v=lin.Value)

  except KeyboardInterrupt:
    pass

  finally:
    lin.Stop()
