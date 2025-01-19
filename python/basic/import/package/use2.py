#!/usr/bin/python3
#\file    use2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.29, 2015

import os
os.environ['ROBOT']= 'baxter'
#import pkg  #works
from pkg import *  #works

if __name__=='__main__':
  Func1()
  Func2()
