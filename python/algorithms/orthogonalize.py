#!/usr/bin/python
import numpy as np
import numpy.linalg as la

#Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
#original_norm: keep original vec's norm, otherwise the norm is 1.
def Orthogonalize(vec, base, original_norm=True):
  axis= np.cross(base,vec)
  axis= axis / la.norm(axis)
  vec2= np.cross(axis,base)
  if original_norm:  return vec2 / la.norm(vec2) * la.norm(vec)
  else:              return vec2 / la.norm(vec2)

#Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
#original_norm: keep original vec's norm, otherwise the norm is 1.
#Using The Gram-Schmidt process: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def Orthogonalize2(vec, base, original_norm=True):
  base= np.array(base)/la.norm(base)
  vec2= vec - np.dot(vec,base)*base
  if original_norm:  return vec2 / la.norm(vec2) * la.norm(vec)
  else:              return vec2 / la.norm(vec2)

#Get an orthogonal axis of a given axis
def GetOrthogonalAxis(axis):
  axis= np.array(axis)/la.norm(axis)
  if 1.0-abs(axis[2])>=1.0e-6:
    return Orthogonalize2([0.0,0.0,1.0],base=axis,original_norm=False)
  else:
    return [1.0,0.0,0.0]

import random,copy
vec= [2.0*(random.random()-0.5),2.0*(random.random()-0.5),2.0*(random.random()-0.5)]
base= [2.0*(random.random()-0.5),2.0*(random.random()-0.5),2.0*(random.random()-0.5)]
#vec= np.array(vec)/la.norm(vec)
vec2= Orthogonalize(vec,base,original_norm=True)
vec3= Orthogonalize2(vec,base,original_norm=True)

print '======='
print 'vec=',vec
print 'base=',base
print '-------'
print 'norm(vec)=',la.norm(vec)
print 'dot(vec,base)=',np.dot(vec,base)
print '-------'
print 'vec2=',vec2
print 'norm(vec2)=',la.norm(vec2)
print 'dot(vec2,base)=',np.dot(vec2,base)
print '-------'
print 'vec3=',vec3
print 'norm(vec3)=',la.norm(vec3)
print 'dot(vec3,base)=',np.dot(vec3,base)
print '======='

ex= [2.0*(random.random()-0.5),2.0*(random.random()-0.5),2.0*(random.random()-0.5)]
ex= [1,1,1]
ex= np.array(ex)/la.norm(ex)
ez= GetOrthogonalAxis(ex)
ey= np.cross(ez,ex)
print 'ex=',ex
print 'ey=',ey
print 'ez=',ez
print 'dot(ex,ey)=',np.dot(ex,ey)
print 'dot(ex,ez)=',np.dot(ex,ez)
print 'dot(ey,ez)=',np.dot(ey,ez)
print '======='
