#!/usr/bin/python3
#\file    natural_sort.py
#\brief   List and natural sort files.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.04, 2025
import os,re

'''
Setup:
$ mkdir /tmp/sort
$ touch /tmp/sort/{user.yaml,user2.yaml,user3.yaml,user04.yaml,user10.yaml}
'''

#List files whose names are [prefix][index][suffix] in dir_path,
# and sort them by index in ascending order.
# [index] can be '' which is sorted to the beginning.
def ListAndSortIndexFiles(dir_path, prefix, suffix):
  pattern= rf'{re.escape(prefix)}([\d ]*){re.escape(suffix)}'
  matched_files= [re.fullmatch(pattern, f) for f in os.listdir(dir_path)]
  matched_files= sorted([(m.group(0),m.group(1)) for m in matched_files if m],
                        key=lambda m: -1 if m[1]=='' else int(m[1].strip()))
  return [f for f,idx in matched_files]

if __name__=='__main__':
  print(ListAndSortIndexFiles('/tmp/sort', 'user', '.yaml'))
