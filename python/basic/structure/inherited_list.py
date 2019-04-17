#!/usr/bin/python


class TKeys(list):
  #Ver1: NOTE: this is the best way to wrap a general list
  #def __init__(self,v):
    #list.__init__(self,v)
  #Ver2: NOTE: usage is not intuitive (see examples below)
  #def __init__(self,*v):
    #list.__init__(self,v)
  #Ver3: NOTE: this will fail to create like [[1,2]]; only works with a flat (1-depth) list
  def __init__(self,*v):
    if len(v)==1 and isinstance(v[0],list):
      list.__init__(self,*v)
    else:
      list.__init__(self,v)

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))
  a= TKeys(['a','b','c'])  #Ver1, Ver3
  #a= TKeys('a','b','c')  #Ver2, Ver3
  b= ['c','d','e']
  PrintEq('a')
  PrintEq('b')
  PrintEq('isinstance(a,list)')
  PrintEq('isinstance(a,TKeys)')
  PrintEq('isinstance(b,list)')
  PrintEq('isinstance(b,TKeys)')

  PrintEq('a+b')
  PrintEq('isinstance(a+b,list)')
  PrintEq('isinstance(a+b,TKeys)')

  c= TKeys(a+b)  #Ver1, Ver3
  #c= TKeys(*(a+b))  #Ver2, Ver3
  PrintEq('c')
  PrintEq('isinstance(c,list)')
  PrintEq('isinstance(c,TKeys)')

  PrintEq('c==a+b')

  d= TKeys(b)  #Ver1, Ver3
  #d= TKeys(*b)  #Ver2, Ver3
  PrintEq('b')
  PrintEq('d')
  b[1]= 'aaa'
  PrintEq('b')
  PrintEq('d')
