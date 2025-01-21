#!/usr/bin/python3
import time

time.localtime()
print(time.localtime())
print(time.localtime().tm_sec)

now= time.time()
print(now)
for r in range(10):
  print(time.time()-now)
  time.sleep(0.2)
