#!/usr/bin/python3
#\file    operators.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.03, 2017
import inspect

class Expression(object):
  @staticmethod
  def f():
    pass

class Value(object):
  Data= None

  def __repr__(self):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    return data.__repr__()
  def __str__(self):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    return data.__str__()

  def __lt__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__lt__(other2)
  def __le__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__le__(other2)
  def __eq__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__eq__(other2)
  def __ne__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__ne__(other2)
  def __gt__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__gt__(other2)
  def __ge__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__ge__(other2)

  def __add__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__add__(other2)
  def __sub__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__sub__(other2)
  def __mul__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__mul__(other2)
  def __floordiv__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__floordiv__(other2)
  def __mod__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__mod__(other2)
  def __divmod__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__divmod__(other2)
  #def __pow__(self, other[, modulo]):
  def __lshift__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__lshift__(other2)
  def __rshift__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__rshift__(other2)
  def __and__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__and__(other2)
  def __xor__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__xor__(other2)
  def __or__(self, other):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    other2= other.Data.f() if IsSubCI(other.Data,Expression) else other.Data
    return data.__or__(other2)

  def __hash__(self):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    return data.__hash__()

  def __getattr__(self, name):
    data= self.Data.f() if IsSubCI(self.Data,Expression) else self.Data
    return data.__getattr__(name)

#Check if thing is a subclass or an instance of super class sup_class.
def IsSubCI(thing, sup_class):
  if inspect.isclass(thing):  return issubclass(thing, sup_class)
  return isinstance(thing, sup_class)

Memory= {'b1': 1.5}
class Calc1(Expression):
  @staticmethod
  def f():
    return Memory['b1']

class Position(Value):
  pass
class Position1(Position):
  Data= 1.2
class Position2(Position):
  Data= Calc1


if __name__=='__main__':
  def Print(e,g=globals()):  print(e,'=',eval(e,g))
  def PrintX(e,g=globals()):  print('exec:',e);exec(e,g)

  PrintX('a= Position1()')
  PrintX('b= Position2()')
  Print('a>b')
  Print('a<=b')
  Print('a+b')
  Print('a-b')

