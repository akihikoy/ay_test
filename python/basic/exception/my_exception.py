#!/usr/bin/python3
#\file    my_exception.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.16, 2016
from except_forward import PrintException

class MyException(Exception):
  def __init__(self, msg):
    self.msg= msg
  def __str__(self):
    return repr(self.msg)
  def __repr__(self):
    return 'MyException({msg})'.format(msg=repr(self.msg))

def BadFunc1():
  print('BadFunc1.start')
  raise Exception('BadFunc1 Exception')
  print('BadFunc1.end')

def BadFunc2():
  print('BadFunc2.start')
  raise ValueError('BadFunc2 ValueError')
  print('BadFunc2.end')

def BadFunc3():
  print('BadFunc3.start')
  raise MyException('BadFunc3 MyException')
  print('BadFunc3.end')

def Executor1(func):
  try:
    func()
  except Exception as e:
    PrintException(e, ' ### in Executor1 ###')

def Executor2(func):
  try:
    func()
  except MyException as e:
    PrintException(e, ' ### in Executor2 ###')

if __name__=='__main__':
  print('Executor1...')
  for func in (BadFunc1,BadFunc2,BadFunc3):
    try:
      Executor1(func)
    except Exception as e:
      PrintException(e, ' ### in MAIN ###')

  print('\n-------\n')

  print('Executor2...')
  for func in (BadFunc1,BadFunc2,BadFunc3):
    try:
      Executor2(func)
    except Exception as e:
      PrintException(e, ' ### in MAIN ###')

