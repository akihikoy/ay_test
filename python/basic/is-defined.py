#!/usr/bin/python

print locals()
print 'a is defined' if 'a' in locals() else 'a is not defined'
a= 100
print locals()
print 'a is defined' if 'a' in locals() else 'a is not defined'

class Test:
  def __init__(self):
    #print locals()
    print self.__dict__
    print 'b is defined' if 'b' in self.__dict__ else 'b is not defined'
    self.b= 100
    #print locals()
    print self.__dict__
    print 'b is defined' if 'b' in self.__dict__ else 'b is not defined'
    c= 100
    print self.__dict__
    print locals()

t= Test()
