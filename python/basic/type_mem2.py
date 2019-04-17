#!/usr/bin/python
#\file    type_mem2.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.25, 2017
import string,random
import os,time
from subprocess import call

class NS1(object):
  class C1(object):
    def __init__(self):
      self.X= [i for i in xrange(5000000)]

class NS2(object):
  class C1(object):
    pass

def UniqueID(prefix,N=6):
  for i in xrange(10000000):
    suffix= ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in xrange(N))
    if prefix+suffix not in globals():
      return prefix+suffix

def Add(sup_class=NS1.C1, class_id=None, namespace=NS1):
  if class_id is None:  class_id= UniqueID(sup_class.__name__+'_')
  class_inst= sup_class()
  if isinstance(namespace,dict):
    namespace[class_id]= class_inst
  else:
    setattr(namespace,class_id,class_inst)
  return class_inst

def Del(class_id=None, namespace=NS1):
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
  PrintX('Add(class_id="C2")')
  PrintD('NS1')
  Stat()
  #PrintX('NS1.C2.X= None')  #NOTE: This doesn't matter
  PrintX('Del(class_id="C2")')
  PrintD('NS1')
  Stat()
  #PrintX('Add(class_id="C2")')
  PrintX('Add(class_id="C2")')
  PrintD('NS1')
  Stat()
