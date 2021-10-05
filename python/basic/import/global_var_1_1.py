#!/usr/bin/python
#\file    global_var_1_1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.05, 2021

global_var= 120

class Test(object):
  def __init__(self, var=None):
    global global_var
    if var is None:  var= global_var
    print 'class Test:', var

if __name__=='__main__':
  t1= Test()
  global_var= 100
  t2= Test()
