#!/usr/bin/python

def DictToSelf(class_var, d):
  for key,val in d.iteritems():
    class_var.__dict__[key]= val

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in d_new.iteritems():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new


class TTest:
  def __init__(self):
    self.A= 1.0
    self.B= 2.0
  def Init(self, d):
    DictToSelf(self, d)

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  test= TTest()

  PrintEq('test.__dict__')

  #test.Init({'hoge':2.0, 'aaa':-1.0})
  InsertDict(test.__dict__, {'hoge':2.0, 'aaa':-1.0})

  PrintEq('test.__dict__')
  PrintEq('test.A')
  PrintEq('test.B')
  PrintEq('test.hoge')
  PrintEq('test.aaa')

