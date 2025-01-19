#!/usr/bin/python3
#\file    sed.py
#\brief   sed like function.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.05, 2020
import re

def Sed(filein,fileout,search_expr,replacing):
  with open(filein,'r') as fp:
    lines= fp.readlines()
  with open(fileout,'w') as fp:
    for line in lines:
      fp.write(re.sub(search_expr, replacing, line))

if __name__=='__main__':
  Sed('sed.py','/dev/stdout',r'\bdef\b','DEF')

