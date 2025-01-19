#!/usr/bin/python3
import yaml

#Print a dictionary with a nice format
def PrintDict(d,indent=0):
  for k,v in list(d.items()):
    if type(v)==dict:
      print('  '*indent,'[',k,']=...')
      PrintDict(v,indent+1)
    else:
      print('  '*indent,'[',k,']=',v)
  if indent==0: print('')

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in list(d_new.items()):
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

data= [[1,{'a':2,'b':3}],[0,{'c':2,'b':3}]]

print('data=')
print(data)

yaml_str= yaml.dump(data)
print('yaml_str=')
print(yaml_str)

yaml_dat= yaml.load(yaml_str, Loader=yaml.SafeLoader)
print('yaml_dat=')
print(yaml_dat,'\n')

