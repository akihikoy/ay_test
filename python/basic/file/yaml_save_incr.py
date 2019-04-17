#!/usr/bin/python
#\file    yaml_save_incr.py
#\brief   Save a list as YAML incrementally.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.25, 2017
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

def Main():
  database= []
  fp= open('/tmp/data1.yaml','w')
  for i in range(10):
    data= {}
    data['a']= [i,i*i,i*i*i]
    data['b']= [str(i),str(i*i)]

    database.append(data)
    fp.write(yamldump([data], Dumper=Dumper))
    fp.flush()
  fp.close()

  open('/tmp/data2.yaml','w').write(yamldump(database, Dumper=Dumper))

if __name__=='__main__':
  Main()
