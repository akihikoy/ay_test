#!/usr/bin/python3
#\file    kbhit5.py
#\brief   Testing FlushIn of TKBHit.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.15, 2024
import sys
import time
from kbhit2 import TKBHit

if __name__ == '__main__':
  with TKBHit() as kbhit:
    while 1:
      c= kbhit.KBHit()
      kbhit.FlushIn()
      if c is not None:
        sys.stdout.write(c)
        sys.stdout.flush()
        if c=='q':  break
      else:
        sys.stdout.write('.')
        sys.stdout.flush()

      time.sleep(0.1)

  print('done')

