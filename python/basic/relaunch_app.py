#!/usr/bin/python3
#\file    relaunch_app.py
#\brief   Test of relaunching the program.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.28, 2024
from ask import AskGen
import sys, os

Value= 10

def RelaunchProgram():
  print("Relaunching program...")
  python= sys.executable
  os.execl(python, python, *sys.argv)

if __name__=='__main__':
  flag_relaunch= False
  try:
    while True:
      print('''{}: Command list:
      p: Print and update the state.
      r: Relaunch the program.
      R: Relaunch the program at the end of the program.
      q: Quit the program.
      '''.format(os.getpid()))
      print('{}: Type command >'.format(os.getpid()), end=' ')
      key= AskGen('p','r','q','R')
      if key=='p':
        print('PID={}, Value={}'.format(os.getpid(), Value))
        Value+= 10
      elif key=='r':
        RelaunchProgram()
      elif key=='R':
        flag_relaunch= True
        break
      elif key=='q':
        break

  except Exception as e:
    print('{}: Exception: {}'.format(os.getpid(), e))

  finally:
    print('{}: THE CURRENT PROGRAM IS TO BE TERMINATED#####'.format(os.getpid()))

  if flag_relaunch:
    RelaunchProgram()
