#!/usr/bin/python
import time
t0= time.time()
a=0
for i in range(0,int(1e+6)):
  a+=i
  i+=1
print a, (time.time()-t0)*1.0e3
