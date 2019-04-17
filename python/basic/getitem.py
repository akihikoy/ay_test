#!/usr/bin/python

class TTest:
  def __init__(self):
    self.dic= {}
    self.dic['a']=25.0
    self.dic['b']=5.0
    self.dic['c']=3.0
    self.x= 3.14
  def __getitem__(self,key):
    return self.dic[key]
  def __setitem__(self,key,value):
    self.dic[key]= value


test= TTest()
print "test['a']=",test['a']
print "test['b']=",test['c']
print "test.dic['c']=",test.dic['c']
print "test.x=",test.x

test['a']= 100.0
test['b']= 100.0
test.dic['c']= 100.0
test.x= 100.0

print "test['a']=",test['a']
print "test['b']=",test['c']
print "test.dic['c']=",test.dic['c']
print "test.x=",test.x
