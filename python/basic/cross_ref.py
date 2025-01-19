#!/usr/bin/python3
class TTest:
  def __init__(self):  print('creating:',hex(id(self)))
  def __del__(self):   print('deleting:',hex(id(self)))

def Func1():
  t= TTest()

def Func2():
  t1= TTest()
  t2= TTest()
  t1.t= t2
  t2.t= t1
  #t1.t= None

print('0-----')
Func1()
print('1-----')
Func2()
print('2-----')
