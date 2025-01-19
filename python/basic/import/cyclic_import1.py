#!/usr/bin/python3
#\file    cyclic_import1.py
#\brief   Cyclic import test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.08, 2020

from cyclic_import2 import Func2
#ERROR: ImportError: cannot import name Func2

def Func1():
  print('This is Func1')

if __name__=='__main__':
  print('This is cyclic_import1')
  Func1()
  Func2()
