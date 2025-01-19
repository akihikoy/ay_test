#!/usr/bin/python3
#\file    pyclass2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.24, 2017

#Static variables in a subclass that has multiple superclasses.

class Class1(object):
  prop1= None
  prop2= 2

class Class2(object):
  prop3= 3
  prop4= 4

class Class12_1(Class1,Class2):
  prop1= 1
  prop3= 30
  object= 'x'

if __name__=='__main__':
  def Print(e,g=globals()):  print(e,'=',eval(e,g))
  def PrintX(e,g=globals()):  print('exec:',e);exec(e,g)
  def PrintD(c,k,g=globals()):
    for key in k:
      try:
        print('%s.%s= %r'%(c,key,getattr(eval(c,g),key)))
      except AttributeError:
        print('%s.%s= %r'%(c,key,'<Variable not found>'))
      #found= False
      #for sp in eval(c,g).mro():
        #if key in sp.__dict__:
          #print '%s.%s= %r'%(c,key,sp.__dict__[key])
          #found= True
          #break
      #if not found:  print '%s.%s= %r'%(c,key,'<Variable not found>')
  Print('Class1.mro()')
  Print('Class2.mro()')
  Print('Class12_1.mro()')
  PrintD('Class1',['prop1','prop2','prop3','prop4','object'])
  PrintD('Class2',['prop1','prop2','prop3','prop4','object'])
  PrintD('Class12_1',['prop1','prop2','prop3','prop4','object'])

