#!/usr/bin/python3

#src: http://code.activestate.com/recipes/572182-how-to-implement-kbhit-on-linux/

import sys, termios, atexit
from select import select

# save the terminal settings
fd = sys.stdin.fileno()
new_term = termios.tcgetattr(fd)
old_term = termios.tcgetattr(fd)

# new terminal setting unbuffered
new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)

# switch to normal terminal
def set_normal_term():
  termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

# switch to unbuffered terminal
def set_curses_term():
  termios.tcsetattr(fd, termios.TCSAFLUSH, new_term)

def putch(ch):
  sys.stdout.write(ch)
  sys.stdout.flush()

def getch():
  return sys.stdin.read(1)

def getche():
  ch = getch()
  putch(ch)
  return ch

def kbhit():
  dr,dw,de = select([sys.stdin], [], [], 0)
  #print 'dr:',dr
  #print 'dw:',dw
  #print 'de:',de
  return dr != []

if __name__ == '__main__':
  atexit.register(set_normal_term)
  set_curses_term()

  import time
  disp= '.'
  while 1:
    if kbhit():
      ch= getch()
      sys.stdout.write('> %s\n'%ch)
      sys.stdout.flush()
      while kbhit():
        ch= getch()  #Get the last one of buffer
        sys.stdout.write('>> %s\n'%ch)
        sys.stdout.flush()
      if ch=='q':  break
      else:  disp= ch
      #while kbhit(): getch()  #Clear buffer
    else:
      sys.stdout.write('no kbhit\n')
      while kbhit():
        ch= getch()  #Get the last one of buffer
        sys.stdout.write('>>>> %s\n'%ch)
        sys.stdout.flush()

    for i in range(40):
      sys.stdout.write(disp)
      sys.stdout.flush()
      time.sleep(0.05)
    sys.stdout.write('\n')

  print('done')

