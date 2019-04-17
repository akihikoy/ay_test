#!/usr/bin/python
#\file    arg_expr.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.14, 2016
import sys, random, math

if __name__=='__main__':
  s_expr= sys.argv[1]
  print 'arg[1]=',s_expr
  expr= eval('lambda x:'+s_expr)
  print 'expr=',expr
  for i in range(10):
    x= random.random()
    print 'expr({x})= {value}'.format(x=x, value=expr(x))
