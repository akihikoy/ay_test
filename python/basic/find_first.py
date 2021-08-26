#!/usr/bin/python
#\file    find_first.py
#\brief   Find the first item in a list that matches with a condition.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.26, 2021

if __name__=='__main__':
  l= range(10)
  print next(i for i in l if i>0 and i%3==0)

