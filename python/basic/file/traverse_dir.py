#!/usr/bin/python
#\file    traverse_dir.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.03, 2024
import os

#Traverse a directory dirname including sub-directories and find files that match the condition filepattern.
def TraverseDir(dirname, filepattern=lambda f:os.path.isfile(f)):
  def traverse_dir(dirname):
    return sum([[os.path.join(dirname,f)] if filepattern(os.path.join(dirname,f)) else
                traverse_dir(os.path.join(dirname,f)) if os.path.isdir(os.path.join(dirname,f)) else
                [] for f in os.listdir(dirname)], [])
  dirname= dirname if dirname.endswith('/') else dirname+'/'
  return sorted(map(lambda f:f.replace(dirname,''), traverse_dir(dirname)))

if __name__=='__main__':
  dirname= os.path.join(os.environ['HOME'],'data')
  print 'In dirname, .dat files are:'
  for f in TraverseDir(dirname, filepattern=lambda f:f.endswith('.dat')):
    print f
  print ''
  print 'In dirname, .bag files are:'
  for f in TraverseDir(dirname, filepattern=lambda f:f.endswith('.bag')):
    print f
