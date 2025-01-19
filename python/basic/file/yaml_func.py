#!/usr/bin/python3
import yaml

#Print a dictionary with a nice format
def PrintDict(d,max_level=-1,level=0):
  for k,v in list(d.items()):
    if type(v)==dict:
      print('  '*level,'[',k,']=...')
      if max_level<0 or level<max_level:
        PrintDict(v,max_level,level+1)
    else:
      print('  '*level,'[',k,']=',v)

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in list(d_new.items()):
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

#Load a YAML and insert the data into a dictionary
def InsertYAML(d_base, file_name):
  d_new= yaml.load(file(file_name).read(), Loader=yaml.Loader)
  print(d_new)
  InsertDict(d_base, d_new)


import time,math
def func(x):
  return [x,x,x]

yaml_data='''
help: Container model created by TContainerAnalyzer1.
#Reference AR marker ID and pose:
ref_marker_id: 4
#Pouring edge point candidates:
l_p_pour_e_set:
- [-0.0474839, 0.014636, 0.106539]
#FUNCTION TEST
func_res: !!python/object/apply:yaml_func.func [3.14]
'''

attrib={}
attrib['b1']={}
InsertDict(attrib['b1'], yaml.load(yaml_data, Loader=yaml.Loader))

print('attrib=')
PrintDict(attrib)

