#!/usr/bin/python3
import sys
from kbhit2 import TKBHit

def KBHit():
  kbhit= TKBHit()
  return kbhit.KBHit(timeout=0.1)

if __name__=='__main__':

  import time
  disp= '.'
  while 1:
    c= KBHit()
    print(c)
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

  print('done')


