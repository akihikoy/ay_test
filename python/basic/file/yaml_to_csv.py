#!/usr/bin/python
#\file    yaml_to_csv.py
#\brief   Convert YAML to a CVS.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.25, 2017
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
  from yaml import Loader, Dumper

file_name= '/tmp/log.yaml'
data= yamlload(open(file_name,'r').read(), Loader=Loader)
key_to_file= lambda key: file_name+'.'+key

def Main():
  for key, values in data.iteritems():
    print 'Found:',key
    if not isinstance(values,list):
      print '  non-list value:',values
      continue
    fp= open(key_to_file(key),'w')
    for (tm, v) in values:
      if isinstance(v,list):  fp.write('%f %s\n'%(tm, ' '.join(map(str,v))))
      else:                   fp.write('%f %f\n'%(tm, v))
    fp.close()

def PlotGraphs():
  print 'Plotting graphs..'
  import os
  commands= ['''qplot -x2 aaa {f} w l &'''.format(f=key_to_file(key)) for key in data.keys()]
  commands=[
    '''qplot -x2 aaa -s 'set yrange [0:*]' {f} w l &'''.format(f=key_to_file('area')),
    '''qplot -x2 aaa -s 'set yrange [0:*]' {f1} w l {f2} w l &'''.format(f1=key_to_file('slip'), f2=key_to_file('slip_nml')),
    '''qplot -x2 aaa {f} u 1:2 w l {f} u 1:3 w l &'''.format(f=key_to_file('center')),
    '''qplot -x2 aaa {f} w l &'''.format(f=key_to_file('g_pos')),
    '''qplot -x2 aaa {f1} u 1:4 w l {f2} w l &'''.format(f1=key_to_file('x'), f2=key_to_file('z_trg')),
    '''qplot -x2 aaa {f} u 1:3 w l {f} u 1:6 w l &'''.format(f=key_to_file('f')),
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print '###',cmd
      os.system(cmd)

  print '##########################'
  print '###Press enter to close###'
  print '##########################'
  raw_input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
