#!/usr/bin/python3
#\file    cyclic_import2.py
#\brief   Cyclic import test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.08, 2020

from cyclic_import1 import Func1
#ERROR: ImportError: cannot import name Func1

def Func2():
  print('This is Func2')

if __name__=='__main__':
  print('This is cyclic_import2')
  Func1()
  Func2()
