#!/usr/bin/python
#\file    lib.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.28, 2015

def Run(t, *args):
  x= args[0] if len(args)>0 else None
  y= args[1] if len(args)>1 else None
  print '+++++++++'
  print 'X is:',x
  print 'Y is:',y
  print 'cwd:',os.getcwd()
  print '__package__:',__package__
  print '__file__:',__file__
