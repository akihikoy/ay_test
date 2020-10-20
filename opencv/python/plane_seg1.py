#!/usr/bin/python
#\file    plane_seg1.py
#\brief   Plane segment finder.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.20, 2020
'''
The idea is based on "Fast Coarse Segmentation" of:
https://ieeexplore.ieee.org/abstract/document/6907776
This document is also referred:
https://qiita.com/Kuroyanagi96/items/05c52085f3e67753798a
'''

import numpy as np
import six.moves.cPickle as pickle
import copy
import cv2
import scipy.ndimage
import time
#import matplotlib.pyplot as plt
from pca2 import TPCA,TPCA_SVD
TPCA=TPCA_SVD
#from sklearn.decomposition import PCA

def Normal(img_patch):
  h,w= img_patch.shape[:2]
  points= [[x-w/2,y-h/2,img_patch[y,x]] for y in range(h) for x in range(w) if img_patch[y,x]!=0]
  if len(points)<3:  return None
  normal= TPCA(points).EVecs[:,-1]
  #pca2= PCA(n_components=3,svd_solver='full')  #svd_solver='randomized'
  #pca2.fit(points)
  #normal= pca2.components_[-1]
  if normal[2]<0:  normal= -normal
  return normal

def AvrDepth(img_patch):
  img_valid= img_patch[img_patch!=0]
  if len(img_valid)==0:    return None
  return np.array([np.mean(img_valid)])

def AvrDepth2(img_patch):
  h,w= img_patch.shape[:2]
  points= [[x-w/2,y-h/2,img_patch[y,x]] for y in range(h) for x in range(w) if img_patch[y,x]!=0]
  if len(points)==0:  return None
  return np.array([np.mean(points,axis=0)[-1]])

def ClusteringByFeatures(img, w_patch, f_feat=Normal, th_feat=15):
  img= img.reshape((img.shape[0],img.shape[1]))
  class TNode:
    def __init__(self, patches=[], feat=None):
      self.patches= patches  #patch=(x,y,w,h); (x,y):top-left point
      self.feat= feat
      self.neighbors= set()  #Neighbor nodes that are candidate to be merged.
  #depth image --> patches (w_patch x w_patch); each patch has a feat.
  #patches --> nodes = initial planes
  #nodes= [TNode([(x,y,w_patch,w_patch)],f_feat(img[y:y+w_patch,x:x+w_patch]))
          #for y in range(0,img.shape[0],w_patch)
          #for x in range(0,img.shape[1],w_patch)]
  Nu,Nv= img.shape[1]/w_patch,img.shape[0]/w_patch
  nodes= [TNode([(u*w_patch,v*w_patch,w_patch,w_patch)],f_feat(img[v*w_patch:v*w_patch+w_patch,u*w_patch:u*w_patch+w_patch]))
          for v in range(Nv)
          for u in range(Nu)]
  print Nu,Nv
  #neighbors= ((1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1))
  neighbors= ((1,0),(0,1),(0,-1),(-1,0))
  for node in nodes:
    if node.feat is None:  continue
    x,y,w,h= node.patches[0]
    u,v= x/w,y/h
    node.neighbors= {nodes[(v+dv)*Nu+(u+du)]
                     for du,dv in neighbors
                     if 0<=u+du<Nu and 0<=v+dv<Nv and nodes[(v+dv)*Nu+(u+du)].feat is not None}
  nodes= filter(lambda node:node.feat is not None, nodes)
  debug_id2idx= {node:i for i,node in enumerate(nodes)}

  #Clustering nodes.
  #def remove_node_from_nodes(n):
    #if n in nodes:  nodes.remove(n)
    #for n2 in nodes:
      #if n in n2.neighbors:
  clusters= []
  while nodes:
    #print '#######'
    #print 'nodes',[debug_id2idx[n] for n in nodes]
    #for n2 in nodes:  print '  ',debug_id2idx[n2],'.neighbors',[debug_id2idx[n] for n in n2.neighbors]
    node= nodes.pop( np.random.randint(len(nodes)) )
    #print 'popped',debug_id2idx[node]
    neighbors= node.neighbors
    #print 'neighbors',[debug_id2idx[n] for n in neighbors]
    node.neighbors= set()
    for node2 in neighbors:
      #print '--node2',debug_id2idx[node2]
      #print '--node2.neighbors',[debug_id2idx[n] for n in node2.neighbors]
      node2.neighbors.discard(node)
      #if len(node2.neighbors)==0:
        #if node2 in nodes:  nodes.remove(node2)
      #print '--feat-diff',np.linalg.norm(node.feat-node2.feat),th_feat
      if np.linalg.norm(node.feat-node2.feat) < th_feat:
        #print '--merged',debug_id2idx[node2],'-->',debug_id2idx[node]
        #Merge node2 into node:
        r1= float(len(node.patches))/(len(node.patches)+len(node2.patches))
        node.feat= r1*node.feat + (1.0-r1)*node2.feat
        node.patches+= node2.patches
        node.neighbors.update(node2.neighbors-neighbors)
        #print '--node,node.neighbors',debug_id2idx[node],[debug_id2idx[n] for n in node.neighbors]
        if node2 in nodes:  nodes.remove(node2)
        for node3 in node2.neighbors:
          #if node2 in node3.neighbors:
          node3.neighbors.discard(node2)
        for node3 in node2.neighbors-neighbors:
          node3.neighbors.add(node)
        #DEBUG:
        for node3 in nodes:
          if node2 in node3.neighbors:
            print '>>nodes',[debug_id2idx[n] for n in nodes]
            for n2 in nodes:  print '  ',debug_id2idx[n2],'.neighbors',[debug_id2idx[n] for n in n2.neighbors]
            raise Exception('ERROR',debug_id2idx[node2])
    #print '--',debug_id2idx[node],len(node.neighbors)
    if node.neighbors:
      nodes.append(node)
    else:
      clusters.append(node)
    #print len(nodes)
  return clusters

def DrawClusters(img, clusters):
  img_viz= cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  col_set= ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255))
  for i,node in enumerate(clusters):
    print i, node.feat, len(node.patches)
    col= col_set[i%len(col_set)]
    for patch in node.patches:
      x,y,w,h= patch
      cv2.circle(img_viz, (x+w/2,y+h/2), (w+h)/4, col, 1)
  return img_viz


if __name__=='__main__':
  #img_depth= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth']
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth1.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth001.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth002.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth003.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth004.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  print img_depth.shape, img_depth.dtype, [np.min(img_depth), np.max(img_depth)]

  t_start= time.time()
  clusters= ClusteringByFeatures(img_depth, w_patch=30, th_feat=0.1)
  clusters= [node for node in clusters if len(node.patches)>5]
  print 'Number of clusters:',len(clusters)
  print 'Sum of numbers of patches:',sum([len(node.patches) for node in clusters])
  print 'Computation time:',time.time()-t_start
  img_viz= DrawClusters(img_depth, clusters)

  cv2.imshow('depth',img_viz)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

