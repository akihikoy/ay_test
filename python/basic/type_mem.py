#!/usr/bin/python
#\file    type_mem.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2017
import string,random
import os,time
from subprocess import call

class NS1(object):
  class C1(object):
    X= [i for i in xrange(5000000)]

class NS2(object):
  class C1(object):
    pass

def UniqueID(prefix,N=6):
  for i in xrange(10000000):
    suffix= ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in xrange(N))
    if prefix+suffix not in globals():
      return prefix+suffix

def Add(sup_class=NS1.C1, data={}, class_id=None, namespace=NS1):
  if class_id is None:  class_id= UniqueID(sup_class.__name__+'_')
  class_type= type(class_id, (sup_class,), data)
  if isinstance(namespace,dict):
    namespace[class_id]= class_type
  else:
    setattr(namespace,class_id,class_type)
  return class_type

def Del(class_id=None, namespace=NS1):
  if isinstance(namespace,dict):
    del namespace[class_id]
  else:
    delattr(namespace,class_id)

def Del2(class_inst=None, namespace=NS1):
  for vid in (v for v in class_inst.__dict__.keys() if v[0]!='_'):
    delattr(class_inst,vid)
  class_id= class_inst.__name__
  if isinstance(namespace,dict):
    del namespace[class_id]
  else:
    delattr(namespace,class_id)

if __name__=='__main__':
  def PrintX(e,g=globals()):  print 'exec:',e;exec(e,g)
  def PrintD(c,g=globals()):
    d= eval(c,g)
    k= [x for x in dir(d) if x[0]!='_']
    for key in k:
      try:
        print '%s.%s= %r'%(c,key,getattr(d,key))
      except AttributeError:
        print '%s.%s= %r'%(c,key,'<Variable not found>')

  def Stat():
    time.sleep(0.5)
    call(('ps','v',str(os.getpid())))


  PrintD('NS1')
  PrintX('C2=Add(class_id="C2",data={"X":[i for i in xrange(5000000)]})')
  PrintD('NS1')
  Stat()

  #PrintX('NS1.C2.X= None')  #NOTE: This DOES matter
  #PrintX('Del(class_id="C2")')

  #PrintX('NS1.C2.X= None')  #NOTE: This doesn't matter
  PrintX('Del2(class_inst=C2)')

  PrintD('NS1')
  Stat()
  #PrintX('Add(class_id="C2")')
  PrintX('Add(class_id="C2",data={"X":[i for i in xrange(5000000)]})')
  PrintD('NS1')
  Stat()
