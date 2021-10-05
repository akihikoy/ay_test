#!/usr/bin/python
#\file    global_var_1_2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.05, 2021
from global_var_1_1 import global_var, Test
import global_var_1_1
global_var= 99  #NOTE: This does not modify global_var in global_var_1_1
global_var_1_1.global_var= 98

if __name__=='__main__':
  t1= Test()
  global_var= 100  #NOTE: This does not modify global_var in global_var_1_1
  t2= Test()
  global_var_1_1.global_var= 101
  t3= Test()
