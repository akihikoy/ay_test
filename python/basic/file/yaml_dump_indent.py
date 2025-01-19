#!/usr/bin/python3
#\file    yaml_dump_indent.py
#\brief   Fix the incorrect indentation of lists in YAML.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.8, 2023
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

from yaml import Dumper as yaml_Dumper

class Dumper_IndentPlus(yaml_Dumper):
  def increase_indent(self, flow=False, *args, **kwargs):
    return super(Dumper_IndentPlus,self).increase_indent(flow=flow, indentless=False)

if __name__=='__main__':
  data= {
    'A':{
      'B':[
        {'a':10,'b':20,'c':30},
        {'a':100,'b':200,'c':300},
        {'a':1000,'b':2000,'c':3000},
        ]
      }
    }
  print('Dump with Dumper:')
  print(yamldump(data, Dumper=Dumper))

  print('Dump with Dumper_IndentPlus:')
  print(yamldump(data, Dumper=Dumper_IndentPlus))
