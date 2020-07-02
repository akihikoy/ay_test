#!/usr/bin/python
#\file    recordclass1.py
#\brief   mutable version of namedtuple.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.30, 2020

#NOTE: Need to install recordclass:
#$ pip install recordclass
from recordclass import recordclass

Point= recordclass('Point', ['X','Y'])

a= Point(2.5,-0.5)
a.X= 2.9
a.Y= -0.5

b= Point(0,0)
b.X= 1.1
b.Y= 0.0

print 'a=',a
print 'b=',b
print 'a.X=',a.X
print 'a[1]=',a[1]
print 'list(a)=',list(a)
#print 'a+b=',a+b

