#!/usr/bin/python
#\file    ask.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2017
import sys

def AskYesNo():
  while 1:
    sys.stdout.write('  (y|n) > ')
    ans= sys.stdin.readline().strip()
    if ans=='y' or ans=='Y':  return True
    elif ans=='n' or ans=='N':  return False

#Usage: AskGen('y','n','c')
def AskGen(*argv):
  assert(len(argv)>0)
  while 1:
    sys.stdout.write('  (%s) > ' % '|'.join(argv))
    ans= sys.stdin.readline().strip()
    for a in argv:
      if ans==a:  return a

if __name__=='__main__':
  while True:
    print 'Quit?'
    if AskYesNo():  break
    print 'q:Quit, p:Print, x:Nothing'
    c= AskGen('q','p','x')
    if c=='q':  break
    elif c=='p':  print 'Print xxx'
    elif c=='x':  pass
