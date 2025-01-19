#!/usr/bin/python3
#\file    list_files.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.12, 2020

import os

basedir= '{HOME}/prg/ay_test/python/basic'.format(HOME=os.environ['HOME'])
print('Files in {0}:'.format(basedir), os.listdir(basedir))

for f in os.listdir(basedir):
  print('File:',f ,'isfile?:', os.path.isfile(os.path.join(basedir, f)))
