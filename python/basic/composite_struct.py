#!/usr/bin/python
import numpy as np
import numpy.linalg as la

#Check if a composite vector has a specified structure.
#cstruct is a compared structure that should consist of standard types (int, float)
def CompHasStruct(cvec, cstruct):
  comp_types= (list,tuple,np.ndarray)
  derivables= {}
  derivables[int]= (int,)
  derivables[float]= derivables[int]+(float,np.float_, np.float16, np.float32, np.float64)
  def subcheck(cv,cs):
    if not isinstance(cv,comp_types):
      if cs not in (float,int):  return False
      else:  return type(cv) in derivables[cs]
    else:
      if len(cv)!=len(cs):  return False
      for i in range(len(cs)):
        if not subcheck(cv[i],cs[i]):  return False
      return True
  return subcheck(cvec, cstruct)

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  #p1= [0.1, 0.2]
  #p2= (float,float)
  #p1= [0.1, 0.2]
  #p2= (float,int)
  #p1= 0.1
  #p2= int
  #p1= 99
  #p2= int
  #p1= [1, [1.0, 2.0]]
  #p2= (int,(float,float))
  #p1= [1, [1.0, 2.0]]
  #p2= (float,(float,float))
  #p1= [1, [1.0, 2.0]]
  #p2= (int,(int,float))
  #p1= [1, [1, 2.0]]
  #p2= (int,(int,(float,)))
  p1= [1, [1, [2.0]]]
  p2= (int,(int,(float,)))
  #p1= [1, [1.0, 2.0], 1.0]

  PrintEq('p1')
  PrintEq('p2')
  PrintEq('CompHasStruct(p1, p2)')
