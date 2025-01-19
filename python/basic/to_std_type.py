#!/usr/bin/python3
import numpy as np

def ToStdType(x):
  npbool= (np.bool_)
  npint= (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
  npuint= (np.uint8, np.uint16, np.uint32, np.uint64)
  npfloat= (np.float_, np.float16, np.float32, np.float64)
  if isinstance(x, npbool):   return bool(x)
  if isinstance(x, npint):    return int(x)
  if isinstance(x, npuint):   return int(x)
  if isinstance(x, npfloat):  return float(x)
  if isinstance(x, (int,float,bool,str)):  return x
  if isinstance(x, np.ndarray):  return x.tolist()
  if isinstance(x, (list,tuple)):  return list(map(ToStdType,x))
  if isinstance(x, dict):  return {ToStdType(k):ToStdType(v) for k,v in list(x.items())}
  raise


R= np.array([[1.,0.,0.],[0,1,0],[0,0,1]])
D= {'s':R,'d':[np.array([1,2]),np.array([0.2,0.4])],'c':[np.float64(2.)]}

#npfloat= (np.float_, np.float16, np.float32, np.float64)
#print isinstance(D['c'][0],float)
#print isinstance(D['c'][0],npfloat)
#print ToStdType(D['c'][0])
#print type(ToStdType(D['c'][0]))
#raise

print('R',R)
print('D',D)
print('ToStdType(R)',ToStdType(R))
print('ToStdType(D)',ToStdType(D))

import yaml
#print 'yaml.dump(R)',yaml.dump(R)
#print 'yaml.dump(D)',yaml.dump(D)
print('yaml.dump(ToStdType(R))',yaml.dump(ToStdType(R)))
print('yaml.dump(ToStdType(D))',yaml.dump(ToStdType(D)))
