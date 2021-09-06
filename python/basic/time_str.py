#!/usr/bin/python
#\file    time_str.py
#\brief   Time to string.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.06, 2021
import time

if __name__=='__main__':
  now= time.localtime()
  print 'linux epoch time:', long(time.time()*1e6)
  print 'short:', '%04i%02i%02i%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  print 'short2:', '%04i%02i%02i-%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  print 'normal:', '%04i.%02i.%02i-%02i.%02i.%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
