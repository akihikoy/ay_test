#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from pca2 import TPCA

#Centroid of a polygon
#ref. http://en.wikipedia.org/wiki/Centroid
def PolygonCentroid(points):
  if len(points)==0:  return None
  if len(points)==1:  return points[0]
  assert(len(points[0])==3)
  pca= TPCA(points)
  xy= pca.Projected[:,[0,1]]
  N= len(xy)
  xy= np.vstack((xy,[xy[0]]))  #Extend so that xy[N]==xy[0]
  A= 0.5*sum([xy[n,0]*xy[n+1,1] - xy[n+1,0]*xy[n,1] for n in range(N)])
  Cx= sum([(xy[n,0]+xy[n+1,0])*(xy[n,0]*xy[n+1,1]-xy[n+1,0]*xy[n,1]) for n in range(N)]) / (6.0*A)
  Cy= sum([(xy[n,1]+xy[n+1,1])*(xy[n,0]*xy[n+1,1]-xy[n+1,0]*xy[n,1]) for n in range(N)]) / (6.0*A)
  centroid= pca.Reconstruct([Cx,Cy],[0,1])
  return centroid

from gen_data import *
#points= Gen3d_01()
#points= Gen3d_02()
#points= Gen3d_11()
#points= Gen3d_12()
points= Gen3d_13()

fp= file('/tmp/orig.dat','w')
for p in points:
  fp.write(' '.join(map(str,p))+'\n')
fp.close()

centroid= PolygonCentroid(points)

fp= file('/tmp/res.dat','w')
fp.write(' '.join(map(str,centroid))+'\n')
fp.close()

fp= file('/tmp/res2.dat','w')
fp.write(' '.join(map(str,np.mean(points,axis=0)))+'\n')
fp.close()


