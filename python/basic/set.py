#!/usr/bin/python3
#\file    set.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.20, 2023

if __name__=='__main__':
  s= {1,2,3,4}
  print('s=',s)
  s.add(10)
  print('s=',s)
  s.update({20,21,22})
  print('s=',s)
  s.update([30,31,32])
  print('s=',s)
  s.discard(10)
  print('s=',s)
  s= s-{4,20,30}
  print('s=',s)
  print('list(s)=',list(s))
  print('5 in s=',5 in s)
  print('21 in s=',21 in s)
