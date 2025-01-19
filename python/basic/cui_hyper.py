#!/usr/bin/python3
import readline
import sys,traceback

import numpy as np
import math

histfile = ".pyhist"
try:
  readline.read_history_file(histfile)
except IOError:
  pass

readline.parse_and_bind('tab: complete')

class TCUI:
  def Eval(self,slist):
    return eval(' '.join(slist),globals(),self.__dict__)

  def __init__(self):
    self.lastx=[]

    while True:
      cmd= input('quit or type > ').split()

      try:
        if len(cmd)==0:
          continue
        elif cmd[0] == 'quit' or cmd[0] == 'exit':
          break
        elif cmd[0] == 'home':
          print('home')
        elif cmd[0]=='calc':
          if cmd[1]=='e2q':
            vec= self.Eval(cmd[2:])
            print('Quaternion: ',[2.0*x for x in vec])
          elif cmd[1]=='q2e':
            vec= self.Eval(cmd[2:])
            print('Euler: ',[x/2.0 for x in vec])
          else:
            vec= self.Eval(cmd[1:])
            print('Calc result: ',vec)
        elif cmd[0]=='var':
          vec= self.Eval(cmd[2:])
          self.__dict__[cmd[1]]= vec
          print('New variable ',cmd[1],' : ',self.__dict__[cmd[1]])
        elif cmd[0]=='lastx':
          if len(cmd)==1 or cmd[1]=='show':
            print('Last x: ',self.lastx)
          elif cmd[1]=='set':
            vec= self.Eval(cmd[2:])
            self.lastx= vec
          else:
            print('Invalid lastx-command line: ',' '.join(cmd))
        elif cmd[0]=='move':
          vec= self.Eval(cmd[1:])
          print('move to ',vec[1],' in duration ',vec[0])
        elif cmd[0]=='m':
          vec= self.Eval(cmd[2:])
          print('execute script ',cmd[1],' with arguments ',vec)
        else:
          print('Invalid command line: ',' '.join(cmd))
      except Exception as e:
        print('Error(',type(e),'):')
        print('  ',e)
        #print '  type: ',type(e)
        #print '  args: ',e.args
        #print '  message: ',str(e)
        #print '  sys.exc_info(): ',sys.exc_info()
        print('  Traceback: ')
        traceback.print_tb(sys.exc_info()[2])
        print('Check the command line: ',' '.join(cmd))

TCUI()

readline.write_history_file(histfile)
