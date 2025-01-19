#!/usr/bin/python3
import time
t0= time.time()
a=0
i=0
while i<1000000:
  a+=i
  i+=1
print(a, (time.time()-t0)*1.0e3)
