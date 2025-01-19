#!/usr/bin/python3
#\file    pyclass1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.24, 2017

#Static variables and inheritance.

class Class1(object):
  prop1= None
  prop2= 2

class Class1_1(Class1):
  prop1= 1
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
  #Print('Class1.__dict__')
  #Print('Class1_1.__dict__')
  #Print('Class1().__dict__')
  #Print('Class1_1().__dict__')
  PrintD('Class1',['prop1','prop2','object'])
  PrintD('Class1_1',['prop1','prop2','object'])

  PrintX('Class1_1.prop1= 10')
  PrintD('Class1',['prop1','prop2','object'])
  PrintD('Class1_1',['prop1','prop2','object'])

  PrintX('Class1.prop1= 100')
  PrintD('Class1',['prop1','prop2','object'])
  PrintD('Class1_1',['prop1','prop2','object'])

  PrintX('Class1.prop2= 200')
  PrintD('Class1',['prop1','prop2','object'])
  PrintD('Class1_1',['prop1','prop2','object'])

  Print('Class1_1.object')
  print(Class1_1.object)
