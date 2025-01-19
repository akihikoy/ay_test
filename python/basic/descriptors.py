#!/usr/bin/python3
#\file    descriptors.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.03, 2017
import inspect

class Expression(object):
  @staticmethod
  def f(c):
    pass

class Value(object):
  Data= None
  def __get__(self, instance, owner):
    return self.Data.f(instance) if IsSubCI(self.Data,Expression) else self.Data
  #def __set__(self, instance, value):
  #def __delete__(self, instance)

#Check if thing is a subclass or an instance of super class sup_class.
def IsSubCI(thing, sup_class):
  if inspect.isclass(thing):  return issubclass(thing, sup_class)
  return isinstance(thing, sup_class)

class Calc1(Expression):
  @staticmethod
  def f(c):
    return c.A[1]

class Position(Value):
  pass
class Position1(Position):
  Data= 1.2
class Position2(Position):
  Data= Calc1

class Apple(object):
  Pos= Position1()
class Orange(object):
  A= [1.0,2.0,3.0]
  Pos= Position2()

if __name__=='__main__':
  def Print(e,g=globals()):  print(e,'=',eval(e,g))
  def PrintX(e,g=globals()):  print('exec:',e);exec(e,g)

  PrintX('a= Position1()')
  PrintX('b= Position2()')
  Print('a.Data')
  Print('b.Data')

  PrintX('a= Apple()')
  PrintX('b= Orange()')
  Print('a.Pos')
  Print('b.Pos')
