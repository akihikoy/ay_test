#!/usr/bin/python
#\file    lib1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.28, 2015
import os

def Run(t, *args):
  a= args[0] if len(args)>0 else None
  b= args[1] if len(args)>1 else None
  print '========='
  print 'a is:',a
  print 'b is:',b
  print 'cwd:',os.getcwd()
  print '__package__:',__package__
  print '__file__:',__file__

if __name__=='__main__':
  print 'Executed directly'
  Run(10)
