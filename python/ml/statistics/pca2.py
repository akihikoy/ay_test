#!/usr/bin/python3
#\file    pca2.py
#\brief   PCA with EIG, SVD
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.17, 2016
import numpy as np
import numpy.linalg as la

class TPCA:
  def __init__(self,points,calc_projected=True):
    self.Mean= np.mean(points,axis=0)
    data= points-self.Mean
    cov= np.cov(data.T)
    evals, evecs= la.eig(cov)
    idx= evals.argsort()[::-1]  #Sort by eigenvalue in decreasing order
    self.EVecs= evecs[:,idx]
    self.EVals= evals[idx]
    self.Projected= None
    if calc_projected:
      self.Projected= np.dot(data, self.EVecs)

  def Project(self,points):
    return np.dot(points-self.Mean, self.EVecs)

  def Reconstruct(self,proj,idx=None):
    if idx==None:  idx= list(range(len(self.EVecs)))
    return np.dot(proj, self.EVecs[:,idx].T) + self.Mean


class TPCA_SVD(TPCA):
  def __init__(self,points,calc_projected=True):
    self.Mean= np.mean(points,axis=0)
    data= points-self.Mean
    cov= np.cov(data.T)
    U,S,V= la.svd(cov,0)
    #if S[-1]/S[0]<1.0e-12:  raise Exception('TPCA_SVD: data is singular')
    self.EVecs= V.T
    self.EVals= S
    self.Projected= None
    if calc_projected:
      self.Projected= np.dot(data, self.EVecs)


#Float version of range
def FRange1(x1,x2,num_div):
  return [x1+(x2-x1)*x/float(num_div) for x in range(num_div+1)]

def Main():
  import math
  from random import random
  wrand= 0.05
  c= [-9.99,2.3,0.4]
  r= 0.5
  Rot= np.array([[ 0.9330127 , 0.0669873  ,  0.35355339],
                [ 0.0669873  , 0.9330127  , -0.35355339],
                [-0.35355339 , 0.35355339 ,  0.8660254 ]])
  XYZ=[]
  fp= open('/tmp/data.dat','w')
  #for th in FRange1(0.6*math.pi,0.9*math.pi,100):
  for th in FRange1(-0.2*math.pi,0.9*math.pi,10):
  #for th in FRange1(0.0,2.0*math.pi,50):
    x= r*math.cos(th)+(random()-0.5)*wrand
    y= r*math.sin(th)+(random()-0.5)*wrand
    z= (random()-0.5)*wrand
    xyz= np.array(c)+np.dot(Rot,[x,y,z])
    XYZ.append(xyz.tolist())
    fp.write('%f %f %f\n'%(xyz[0],xyz[1],xyz[2]))
  fp.close()

  fp= open('/tmp/pca_eig.dat','w')
  pca= TPCA(XYZ)
  XY= pca.Projected[:,[0,1]]
  for xy in XY:
    xyz= pca.Reconstruct(xy,[0,1])
    fp.write('%f %f %f\n'%(xyz[0],xyz[1],xyz[2]))
  fp.close()

  fp= open('/tmp/pca_svd.dat','w')
  pca= TPCA_SVD(XYZ)
  XY= pca.Projected[:,[0,1]]
  for xy in XY:
    xyz= pca.Reconstruct(xy,[0,1])
    fp.write('%f %f %f\n'%(xyz[0],xyz[1],xyz[2]))
  fp.close()

def PlotGraphs():
  print('Plotting graphs..')
  import os
  commands=[
    '''qplot -x2 aaa -3d -s 'set size square;set size ratio -1'
          /tmp/data.dat w p pt 6
          /tmp/pca_eig.dat w lp
          /tmp/pca_svd.dat w lp
          &''',
    '''''',
    '''''',
    ]
  for cmd in commands:
    if cmd!='':
      cmd= ' '.join(cmd.splitlines())
      print('###',cmd)
      os.system(cmd)

  print('##########################')
  print('###Press enter to close###')
  print('##########################')
  input()
  os.system('qplot -x2kill aaa')

if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
