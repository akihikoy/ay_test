#!/usr/bin/python
#\file    kbhit4.py
#\brief   Testing "with" version of TKBHit.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.09, 2020
import sys
import time
from kbhit2 import TKBHit

if __name__ == '__main__':
  with TKBHit(activate=True) as kbhit:
    disp= '.'
    while 1:
      c= kbhit.KBHit()
      if c is not None:
        sys.stdout.write('> %r\n'%c)
        sys.stdout.flush()
        if c=='q':  break
        else:  disp= c

      for i in range(40):
        sys.stdout.write(disp)
        sys.stdout.flush()
        time.sleep(0.05)
      sys.stdout.write('\n')

  print 'done'

