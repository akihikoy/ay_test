#!/usr/bin/python
#\file    lib1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.28, 2015
import os

def Run(t, *args):
  x= args[0] if len(args)>0 else None
  y= args[1] if len(args)>1 else None
  print '---------'
  print 'x is:',x
  print 'y is:',y
  print 'cwd:',os.getcwd()
  print '__package__:',__package__
  print '__file__:',__file__
  t.ExecuteMod('lib.sub.lib2', -1)

if __name__=='__main__':
  print 'Executed directly'
  import sys
  sys.path.append('../')
  from load_test import TCore
  t= TCore()
  Run(t,100)
