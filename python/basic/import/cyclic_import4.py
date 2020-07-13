#!/usr/bin/python
#\file    cyclic_import4.py
#\brief   Cyclic import test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.08, 2020

def Func2():
  print 'This is Func2'

from cyclic_import3 import Func1
#ERROR: ImportError: cannot import name Func1

if __name__=='__main__':
  print 'This is cyclic_import4'
  Func1()
  Func2()
