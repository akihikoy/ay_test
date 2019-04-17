#!/usr/bin/python
#\file    pyclass4.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.26, 2017

#Good way of repr(Class)

class Class1(object):
  prop1= None
  prop2= 2
  #@staticmethod
  #def __repr__():
    #return '(%r,%r)'%(Class1.prop1,Class1.prop2)
  @staticmethod
  def Repr(s):
    return '(%r,%r)'%(s.prop1,s.prop2)

class Class1_1(Class1):
  prop1= 1
  object= 'x'

class Class1_2(Class1):
  prop1= 100
  object= 'x'
  @staticmethod
  def Repr(s):
    return '(%r,%r,%r)'%(s.prop1,s.prop2,s.object)

#Return a representation of a class.
def Repr(cls):
  #return repr(cls)
  return cls.Repr(cls)

if __name__=='__main__':
  def Print(e,g=globals()):  print e,'=',eval(e,g)
  def PrintX(e,g=globals()):  print 'exec:',e;exec(e,g)
  def PrintD(c,k,g=globals()):
    for key in k:
      try:
        print '%s.%s= %r'%(c,key,getattr(eval(c,g),key))
      except AttributeError:
        print '%s.%s= %r'%(c,key,'<Variable not found>')
      #found= False
      #for sp in eval(c,g).mro():
        #if key in sp.__dict__:
          #print '%s.%s= %r'%(c,key,sp.__dict__[key])
          #found= True
          #break
      #if not found:  print '%s.%s= %r'%(c,key,'<Variable not found>')
  Print('Repr(Class1)')
  Print('Repr(Class1_1)')
  Print('Repr(Class1_2)')

