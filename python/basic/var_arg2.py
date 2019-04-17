#!/usr/bin/python
#\file    var_arg2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.28, 2018

def Run(ct,*args):
  print 'type of args:', type(args)
  print 'length of args:', len(args)
  print 'content of args:', args

ct= None
Run(ct, 'aaa')
Run(ct, 1, 2.5, 'bbb', [1,2,3])

