#!/usr/bin/python
#\file    yaml_simpledict.py
#\brief   Saving a simple structure of dict in a YAML format; testing default_flow_style parameter of yaml dump.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2024
import os,sys
from yaml import load as yamlload
from yaml import dump as yamldump
try:
  from yaml import CLoader as YLoader, CDumper as YDumper
except ImportError:
  from yaml import Loader as YLoader, Dumper as YDumper
from yaml import Dumper as yaml_Dumper

#Load a dict from a YAML string
def LoadYAMLStr(s):
  wrong_directive= '%YAML:1.0'
  if s.startswith(wrong_directive):  #Skip the first line if it is a wrong directive generated by OpenCV.
    s= s[len(wrong_directive):]
  return yamlload(s, Loader=YLoader)

#Load a YAML file and return a dictionary
def LoadYAML(file_name):
  with open(file_name) as fp:
    text= fp.read()
    return LoadYAMLStr(text)

#Dump a dictionary d in a YAML string.
#If directive (str) is given, it is inserted at the beginning of the YAML.
#If correct_indent, Dumper_IndentPlus is used as the dumper.
#If to_std_type, the input dictionary is converted to regular types by ToStdType with except_cnv.
def DumpYAML(d, except_cnv=lambda y:y, directive=None, correct_indent=True, to_std_type=True):
  s= ''
  if directive is not None:
    s+= directive+'\n'
  #d= ToStdType(d,except_cnv)
  s+= yamldump(d, Dumper=Dumper_IndentPlus if correct_indent else YDumper, default_flow_style=False)
  return s

#Save a dictionary d into a file file_name in YAML format.
#If directive (str) is given, it is inserted at the beginning of the YAML.
#If interactive, prompted before overwriting the file_name.
#If correct_indent, Dumper_IndentPlus is used as the dumper.
#If to_std_type, the input dictionary is converted to regular types by ToStdType with except_cnv.
def SaveYAML(d, file_name, except_cnv=lambda y:y, interactive=False, directive=None, correct_indent=True, to_std_type=True):
  with open(file_name,'w') as fp:
    fp.write(DumpYAML(d, except_cnv=except_cnv, directive=directive, correct_indent=correct_indent, to_std_type=to_std_type))

#Dumper class to correct the list indentation issue of the original Dumper.
class Dumper_IndentPlus(yaml_Dumper):
  def increase_indent(self, flow=False, *args, **kwargs):
    return super(Dumper_IndentPlus,self).increase_indent(flow=flow, indentless=False)

if __name__=='__main__':
  default_flow_style= False

  data=dict(
    aaa=124,
    bbb='bbb',
    ccc=12.5,
    #ddd=[1,2,3],
    )

  yaml_str= yamldump(data, default_flow_style=default_flow_style)
  print 'yaml_str='
  print yaml_str
  SaveYAML(data, '/tmp/test.yaml')
  print 'yaml_str/1.2='
  with open('/tmp/test.yaml') as fp:
    print fp.read()

  #yaml_dat= yamlload(yaml_str)
  #print 'yaml_dat='
  #print yaml_dat,'\n'

  data['eee']= dict(xxx=123,yyy='hoge')
  yaml_str= yamldump(data, default_flow_style=default_flow_style)
  print 'yaml_str/2='
  print yaml_str

  #yaml_dat= yamlload(yaml_str)
  #print 'yaml_dat/2='
  #print yaml_dat,'\n'
