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

attrib={}
attrib['b1']={}
attrib['b1']['g_width']= 0.1
attrib['b1']['p_set']= [[1,2,3], [2,3,4], [1,2,3], [2,3,4]]
attrib['b1']['r_dict']= {}
attrib['b1']['r_dict']['aaa']= 1.25
attrib['b1']['r_dict']['bbb']= 3.14
attrib['b1']['r_dict']['ddd']= [0,0,0]
attrib['b1']['models']=[{'kind':'Sphere','radius':3,'p':[0,0,0]},
                        {'kind':'Cylinder','radius':2,'p1':[0,0,0],'p2':[1,1,0]}]
attrib['b2']={}
attrib['b2']['g_width']= 0.5
attrib['b2']['p_set']= [[1,2,3]]
attrib['b2']['q_set']= [1,2,3]
attrib['b2']['r_dict']= {}
attrib['b2']['r_dict']['aaa']= 0.00
attrib['b2']['r_dict']['ccc']= 9.99

print('attrib=')
PrintDict(attrib)

yaml_b1_str= yaml.dump(attrib['b1'])
print('yaml_b1_str=')
print(yaml_b1_str)

yaml_b1_dat= yaml.load(yaml_b1_str, Loader=yaml.SafeLoader)
print('yaml_b1_dat=')
print(yaml_b1_dat,'\n')

#attrib['b2']= dict(attrib['b2'], **yaml_b1_dat)
InsertDict(attrib['b2'], yaml_b1_dat)

print('attrib=')
PrintDict(attrib)

yaml_str= yaml.dump(attrib)
print('yaml_str=')
print(yaml_str)
