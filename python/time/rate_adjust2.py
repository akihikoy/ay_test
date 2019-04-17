#!/usr/bin/python
#\file    rate_adjust2.py
#\brief   Test of TRate.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.09, 2018

from rate_adjust import TRate

if __name__=='__main__':
  rate= TRate(100.0)
  while True:
    rate.sleep()
