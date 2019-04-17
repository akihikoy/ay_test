#!/usr/bin/python
#\file    iter1.py
#\brief   Test of iterator
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.04, 2017

#We want to read 3 elements of L at each step, but the step size should be 2.
#Expected output is: (0,1,2), (2,3,4), (4,5,6), (6,7,8)
#This would be hard to do with "for".

L= range(10)

try:
  itr= iter(L)
  c= itr.next()
  while True:
    a,b,c= c,itr.next(),itr.next()
    print (a,b,c)
except StopIteration:
  pass

