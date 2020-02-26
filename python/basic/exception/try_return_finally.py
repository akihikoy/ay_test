#!/usr/bin/python
#\file    try_return_finally.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.30, 2020

def func0():
  print 'entered into: func0'
  return 'func0 return'

def func1():
  try:
    return func0()
  finally:  #Finally is executed after the execution of func0
    print 'entered into: func1 finally'
    #and after finally, the return value of func0 is returned.

def func2():
  try:
    return func0()
  finally:
    print 'entered into: func2 finally'
    return 'func2 return'  #This OVERWRITES the return in try.

def func3():
  try:
    return func0()
  finally:
    print 'entered into: func3 finally'
  return 'func3 return'  #This is not executed.

res= func1()
print 'Return of func1:',res
print '----------'

res= func2()
print 'Return of func2:',res
print '----------'

res= func3()
print 'Return of func3:',res
print '----------'

