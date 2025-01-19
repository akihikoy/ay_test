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
  d_new= yaml.load(open(file_name).read(), Loader=yaml.SafeLoader)
  #print d_new
  InsertDict(d_base, d_new)

attrib={}
attrib['*']={}
InsertYAML(attrib['*'], 'data/b50.yaml')
#InsertYAML(attrib['*'], '/home/akihiko/tmp/test20150112-152613.yaml')
#InsertYAML(attrib['*'], '/home/akihiko/ros_ws/pr2_lfd_trick/.memory.yaml')

print('attrib=')
PrintDict(attrib)

