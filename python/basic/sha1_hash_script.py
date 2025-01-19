#!/usr/bin/python3
#\file    sha1_hash_script.py
#\brief   Get SHA1 hash of a script program.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.27, 2021

import hashlib
import os


if __name__=='__main__':
  script_name= os.path.dirname(os.path.abspath(__file__))+'/sha1_hash_dict.py'
  print(hashlib.sha1(open(script_name).read().encode('utf-8')).hexdigest())
