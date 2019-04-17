#!/usr/bin/python
#\file    static_var.py
#\brief   Test of static variable.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2017

def F1(static_var=None):
  print 'static_var=',static_var
  if static_var is None:  static_var= 0
  else:  static_var+= 1

def F2(static_var=[None]):
  print 'static_var=',static_var
  if static_var[0] is None:  static_var[0]= 0
  else:  static_var[0]+= 1

F1()
F1()
F1()
F1()

F2()
F2()
F2()
F2()
