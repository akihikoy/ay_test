#!/usr/bin/python3
#\file    bisect1.py
#\brief   Test of bisect.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.30, 2023
from __future__ import print_function
import bisect
import numpy as np

def RandName(s):
  alph= 'abcdef'
  l= len(alph)
  return alph[s%l]+alph[(s//l)%l]+alph[(s//l//l)%l]+alph[(s//l//l//l)%l]

class TKeyWrapper:
  def __init__(self, iterable, key):
    self.it = iterable
    self.key = key
  def __getitem__(self, i):
    return self.key(self.it[i])
  def __len__(self):
    return len(self.it)

if __name__=='__main__':
  l= sorted(np.random.randint(0,10,size=10))
  print('l= {} (len={})'.format(l,len(l)))
  new= np.random.randint(0,10)
  i_new= bisect.bisect(l,new)
  print('Adding {} at {}'.format(new,i_new))
  l.insert(i_new,new)
  print('l= {} (len={})'.format(l,len(l)))

  #WARNING: key parameter of bisect is added in Python 3.10
  def rand_entry():
    return dict(name=RandName(np.random.randint(0,6**4)),
                height=np.random.randint(100,200),
                age=np.random.randint(0,100))
  d= [rand_entry() for i in range(10)]
  d.sort(key=lambda e:e['height'])
  print('d= {} (len={})'.format(',\n   '.join(map(repr,d)),len(d)))
  new= rand_entry()
  i_new= bisect.bisect(TKeyWrapper(d,key=lambda e:e['height']),new['height'])
  print('Adding {} at {}'.format(new,i_new))
  d.insert(i_new,new)
  print('d= {} (len={})'.format(',\n   '.join(map(repr,d)),len(d)))
