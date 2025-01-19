#!/usr/bin/python3
import collections

Point= collections.namedtuple('Point', ['X','Y'], verbose=True)

a= Point(2.5,-0.5)
#a.X= 2.9
#a.Y= -0.5

b= Point(0,0)
#b.X= 1.1
#b.Y= 0.0

print('a=',a)
print('b=',b)
print('a.X=',a.X)
print('a[1]=',a[1])
print('list(a)=',list(a))
print('a+b=',a+b)


#NOTE: it's impossible to assign to namedtuple object (it is immutable).
#e.g.
#a.X= 2.9
#For this purpose, use: ###recordtype###
# http://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python
#Or, use: ###recordclass###
# https://stackoverflow.com/questions/29290359/existence-of-mutable-named-tuple-in-python
# cf. recordclass1.py

