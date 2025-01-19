#!/usr/bin/python3
import math

class TTest1:
  X= 10

  def __init__(self):
    self.a= 1
    self.b= 100
    self.c= [1,2,3,4]

  def Members(self):
    print(self.__dict__)

  def Print(self,a='aaa'):
    print(a,self.c)

class TTest2:
  X= 10

  def __init__(self):
    self.a= 2
    self.b= 200
    self.c= [1,2,3,4,5]
    self.d= 400

  def Members(self):
    print(self.__dict__)

  def Print(self,a='bbb'):
    print(a,self.c,self.d)

test1= TTest1()
test1.x= 9

print('test1=')
test1.Members()
test1.Print()
#print locals()
#print test1

test2= TTest2()

print('test2=')
test2.Members()
test2.Print()

old_dict= test1.__dict__
#test2.__dict__= old_dict
for k,v in list(old_dict.items()):
  test2.__dict__[k]= v

print('test2=')
test2.Members()
test2.Print()

