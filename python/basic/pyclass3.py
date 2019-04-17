#!/usr/bin/python
#\file    pyclass3.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2017

#Static variable are really shared?

class Class1(object):
  v= None
  l= [1,2,3]

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
  PrintD('Class1',['v','l'])
  PrintX('c1=Class1()')
  PrintD('c1',['v','l'])
  PrintD('Class1',['v','l'])
  PrintX('c1.v= 10')
  PrintX('c1.l[1]*= 10')
  PrintD('c1',['v','l'])
  PrintD('Class1',['v','l'])
  PrintX('c1.l= [3,2,1]')
  PrintD('c1',['v','l'])
  PrintD('Class1',['v','l'])

