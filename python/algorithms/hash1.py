#!/usr/bin/python3
#\file    hash1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.13, 2016

#Hashable (str,int) pair. str is the primary key in sorting.
#Do not modify the content.
class TStrInt(object):
  def __init__(self,s='',i=0):
    self.__s= s
    self.__i= i
  def __key(self):
    return (self.__s,self.__i)
  def __str__(self):
    return str(self.__key())
  def __repr__(self):
    return str(self.__key())
  def __eq__(x, y):
    if type(x)!=type(y):  return False
    return x.__key()==y.__key()
  def __ne__(x, y):
    if type(x)!=type(y):  return True
    return x.__key()!=y.__key()
  def __gt__(x, y):
    return x.__key()>y.__key()
  def __ge__(x, y):
    return x.__key()>=y.__key()
  def __lt__(x, y):
    return x.__key()<y.__key()
  def __le__(x, y):
    return x.__key()<=y.__key()
  def __hash__(self):
    return hash(self.__key())
  def __add__(x, y):
    if isinstance(y,int):  return TStrInt(x.__s,x.__i+y)
    if isinstance(y,TStrInt):  return TStrInt(x.__s+y.__s,x.__i+y.__i)
    raise TypeError('Not defined operation: TStrInt + %s'%type(y))
  def __sub__(x, y):
    if isinstance(y,int):  return TStrInt(x.__s,x.__i-y)
    raise TypeError('Not defined operation: TStrInt - %s'%type(y))
  @property
  def S(self):
    return self.__s
  @property
  def I(self):
    return self.__i

if __name__=='__main__':
  def Print(*ss):
    for s in ss:
      print(s, end=' ')
    print('')
  a= TStrInt('a',0)
  test= lambda s: Print(s,'=',eval(s))
  test('''a''')
  print('')
  test('''a==TStrInt('a',0)''')
  test('''a==TStrInt('a',1)''')
  test('''a!=TStrInt('a',0)''')
  test('''a!=TStrInt('a',1)''')
  print('')
  test('''a>TStrInt('a',0)''')
  test('''a<TStrInt('a',0)''')
  test('''a>TStrInt('a',1)''')
  test('''a<TStrInt('a',1)''')
  print('')
  test('''a>=TStrInt('a',0)''')
  test('''a<=TStrInt('a',0)''')
  test('''a>=TStrInt('a',1)''')
  test('''a<=TStrInt('a',1)''')
  print('')

  x= {}
  x[TStrInt('a',1)]= [1,2,3]
  x[TStrInt('a',0)]= [0,0,0]
  x[TStrInt('b',0)]= [4,5,6]
  test('''x''')
  print('')

  test('''a+TStrInt('a',1)''')
  test('''a+1''')
  test('''a-2''')
  test('''a-2==TStrInt('a',-1)''')
  test('''a-2==TStrInt('a',-2)''')
  print('')

  test('''a''')
  test('''a.S''')
  test('''a.I''')
  #a.S= 'b'
  #a.I= 10
  test('''a''')
  print('')
