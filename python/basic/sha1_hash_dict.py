#!/usr/bin/python3
#\file    sha1_hash_dict.py
#\brief   Get SHA1 hash of a dictionary content;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.27, 2021
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

import hashlib

#Print a dictionary with a nice format
def PrintDict(d,indent=0):
  for k,v in d.items():
    if type(v)==dict:
      print('  '*indent,'[',k,']=...')
      PrintDict(v,indent+1)
    else:
      print('  '*indent,'[',k,']=',v)
  if indent==0: print('')

def GetSHA1HashOfDict(d):
  d_yaml= yamldump(d, Dumper=Dumper)
  return hashlib.sha1(d_yaml.encode('utf-8')).hexdigest()

if __name__=='__main__':
  d1= {
    'a': {
      'a': [0.49, 0.03, -0.194, 0,0,0,1],
      'b': 0.08,
      'c': 'TEST',
      'd': [[-0.11,-0.15,0.11], [-0.11,0.15,0.11], [-0.125,0.15,0.11], [-0.125,0.15,0.15] ],
      'e': {
        'f':{
          'a': [0.0,0.0,0.0, 0.0,0.0,0.0,1.0],
          'b': [[-0.11,-0.15], [-0.11,0.15], [0.085,0.15], [0.085,-0.15] ],
          },
        'g':{
          'a': [0.085,0.0,0.0, -0.0,-0.3420201433256687,-0.0,0.9396926207859084],
          'b': [[0.0,0.15], [0.0,-0.15], [0.20,-0.15], [0.20,0.15] ],
          },
        },
      },
    'b': {
      'a': 20,
      'b': [0.001, 0.15],
      'c': 0.85,
      'd': [-1.5707963267948966, 1.5707963267948966],
      'e': {
        'a': 1.0,
        'b': 0.8,
        'd': 0.2,
        },
      },
    }
  PrintDict(d1)
  print('yamldump(d1):')
  print(yamldump(d1, Dumper=Dumper))

  print('GetSHA1HashOfDict(d1)',GetSHA1HashOfDict(d1))

