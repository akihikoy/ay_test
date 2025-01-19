#!/usr/bin/python
from __future__ import print_function
import xml.dom.minidom

#Print a dictionary with a nice format
def PrintDict(d,indent=0):
  if isinstance(d,dict):
    items= list(d.items())
  elif isinstance(d,(list,tuple)):
    items= list(zip(list(range(len(d))),d))
  for k,v in items:
    if type(v)==dict:
      print('  '*indent,'[',k,']=<dict>...')
      PrintDict(v,indent+1)
    elif k not in ('parentNode','ownerElement','ownerDocument','previousSibling','nextSibling') and getattr(v,'__dict__',None):
      print('  '*indent,'[',k,']=<class>.__dict__...')
      PrintDict(v.__dict__,indent+1)
    elif isinstance(v,(list,tuple)):
      print('  '*indent,'[',k,']=<list>...')
      PrintDict(v,indent+1)
    else:
      print('  '*indent,'[',k,']=',v)
  if indent==0: print('')

def PrintRecursively(dom):
  print('dom.documentElement.tagName:',dom.documentElement.tagName)
  #print 'dom.documentElement has __dict__:',getattr(dom.documentElement,'__dict__',None)
  PrintDict(dom.documentElement.__dict__)
  #print dom,':',dom.__dict__
  #print dom.documentElement.nodeType,':',dom.documentElement.__dict__
  #def sub_proc(nodes,i):
    #for node in nodes:
      ##if node.nodeType == node.ELEMENT_NODE:  # node.TEXT_NODE
        ##print '--'*i,node.tagName
      ##else:
        ##print '--'*i,node.data
      #print '--'*i,node.nodeType,':',node.__dict__
      #if len(node.childNodes)>0:
        #sub_proc(node.childNodes,i+1)
  #sub_proc(dom.documentElement.childNodes,1)

def FindFirst(dom,cond):
  def sub_proc(d,cond):
    if cond(d):  return d
    if isinstance(d,dict):
      items= list(d.items())
    elif isinstance(d,(list,tuple)):
      items= list(zip(list(range(len(d))),d))
    for k,v in items:
      if type(v)==dict:
        res= sub_proc(v,cond)
      elif k not in ('parentNode','ownerElement','ownerDocument','previousSibling','nextSibling') and getattr(v,'__dict__',None):
        res= sub_proc(v.__dict__,cond)
      elif isinstance(v,(list,tuple)):
        res= sub_proc(v,cond)
      else:
        if cond(v):  res= v
        else:  res= None
      if res!=None:  return res
    return None
  return sub_proc(dom.documentElement.__dict__,cond)


if __name__=='__main__':
  dom= xml.dom.minidom.parse("data/robot_description.xml")

  PrintRecursively(dom)

  print("-------------------------------------")

  for url in dom.getElementsByTagName("url"):
    print(url.firstChild.data)

  print("-------------------------------------")

  #find d['_attrs']['name'].value==l_gripper_sensor_mount_joint
  def search_cond(d):
    #return d._attrs.name.value=='l_gripper_sensor_mount_joint'
    res= isinstance(d,dict) and '_attrs' in d and 'name' in d['_attrs'] and getattr(d['_attrs']['name'],'value',None) and d['_attrs']['name'].value=='l_gripper_sensor_mount_joint'
    #res= isinstance(d,dict) and '_attrs' in d and (isinstance(d['_attrs'],dict))
    #if res:  print type(d['_attrs'])
    #if res:  print res
    return res

  node= FindFirst(dom,search_cond)
  if node!=None:
    print('Found.')
    PrintDict(node)

