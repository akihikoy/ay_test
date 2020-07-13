#!/usr/bin/python
#\file    cyclic_import3.py
#\brief   Cyclic import test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.08, 2020

def Func1():
  print 'This is Func1'

from cyclic_import4 import Func2
#ERROR: ImportError: cannot import name Func2

if __name__=='__main__':
  print 'This is cyclic_import3'
  Func1()
  Func2()
