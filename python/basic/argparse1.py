#!/usr/bin/python3
#\file    argparse1.py
#\brief   Test of argparse.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.09, 2022
import argparse
parser= argparse.ArgumentParser(description='Test of argparse.')
addarg= lambda *args,**kwargs: parser.add_argument(*args,**kwargs)
addarg('a1', metavar='A1', type=int, nargs='+',
       help='a1 value')
addarg('-o', '--op', dest='op', default='plus_one',
       help='action')
addarg('-c', default='c_arg',
       help='c_arg')
addarg('-f', action='store_true',
       help='flag')

args= parser.parse_args()
print(('a1=',args.a1))
print(('op=',args.op))
print(('c=',args.c))
print(('f=',args.f))

