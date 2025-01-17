#!/usr/bin/python3
import numpy as np
import numpy.linalg as la

class TPCA:
  def __init__(self,points):
    self.Mean= np.mean(points,axis=0)
    #print 'Mean=',self.Mean
    data= points-self.Mean
    cov= np.cov(data.T)
    #print 'cov=',cov
    evals, evecs= la.eig(cov)
    idx= evals.argsort()[::-1]  #Sort by eigenvalue in decreasing order
    #print 'idx=',idx
    self.EVecs= evecs[:,idx]
    self.EVals= evals[idx]
    #print 'evecs=',self.EVecs
    #print 'evals=',self.EVals
    self.Projected= np.dot(data, self.EVecs)
    #print 'Projected=',self.Projected

  def Reconstruct(self,proj,idx=None):
    if idx==None:  idx= list(range(len(self.EVecs)))
    return np.dot(proj, self.EVecs[:,idx].T) + self.Mean


import yaml
#points= yaml.load(open('data/polygon.yaml').read(), Loader=yaml.SafeLoader)['polygon']
points= yaml.load(open('data/b51.yaml').read(), Loader=yaml.SafeLoader)['l_p_pour_e_set']
#print points

pca= TPCA(points)

fp= open('/tmp/orig.dat','w')
for p in points:
  fp.write(' '.join(map(str,p))+'\n')
fp.close()

#proj= pca.Projected
#proj[:,2]= 0
#rescaled= pca.Reconstruct(proj)

proj= pca.Projected[:,[0,1]]
rescaled= pca.Reconstruct(proj,[0,1])

fp= open('/tmp/res.dat','w')
for p in rescaled:
  fp.write(' '.join(map(str,p))+'\n')
fp.close()

