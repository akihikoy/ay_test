#!/usr/bin/python
import yaml

#Print a dictionary with a nice format
def PrintDict(d,max_level=-1,level=0):
  for k,v in d.items():
    if type(v)==dict:
      print '  '*level,'[',k,']=...'
      if max_level<0 or level<max_level:
        PrintDict(v,max_level,level+1)
    else:
      print '  '*level,'[',k,']=',v

#Insert a new dictionary to the base dictionary
def InsertDict(d_base, d_new):
  for k_new,v_new in d_new.items():
    if k_new in d_base and (type(v_new)==dict and type(d_base[k_new])==dict):
      InsertDict(d_base[k_new], v_new)
    else:
      d_base[k_new]= v_new

#Load a YAML and insert the data into a dictionary
def InsertYAML(d_base, file_name):
  d_new= yaml.load(file(file_name).read())
  print d_new
  InsertDict(d_base, d_new)

attrib={}
attrib['b1']={}
InsertYAML(attrib['b1'], 'data/b50.yaml')
attrib['b2']={}
InsertYAML(attrib['b2'], 'data/b51.yaml')

print 'attrib='
PrintDict(attrib)

#print type(attrib['b2']['bool_test1_1'])
#print type(attrib['b2']['bool_test1_2'])
#print type(attrib['b2']['bool_test1_3'])
#print type(attrib['b2']['bool_test2_1'])
#print type(attrib['b2']['bool_test2_2'])
#print type(attrib['b2']['bool_test2_3'])
