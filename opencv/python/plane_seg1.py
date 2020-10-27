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

#Feature definition for clustering (interface class).
class TImgPatchFeatIF(object):
  #Set up the feature.  Parameters may be added.
  def __init__(self):
    pass
  #Get a feature vector for an image patch img_patch.
  #Return None if it is impossible to get a feature.
  def __call__(self,img_patch):
    return None
  #Get a difference (scholar value) between two features.
  def Diff(self,f1,f2):
    return None
  #Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  def WSum(self,f1,f2,w1):
    return None

#Feature of the normal of a patch.
#  th_normal: The feature is None if the normal length is greater than this value.
class TImgPatchFeatNormal(TImgPatchFeatIF):
  def __init__(self, th_normal=0.4):
    self.th_normal= th_normal
  #Get a normal vector of an image patch img_patch.
  def __call__(self,img_patch):
    h,w= img_patch.shape[:2]
    #NOTE: Change the step of range from 1 to 2 for speed up.
    #points= [[x-w/2,y-h/2,img_patch[y,x]] for y in range(h) for x in range(w) if img_patch[y,x]!=0]
    points= np.vstack([np.where(img_patch!=0), img_patch[img_patch!=0].ravel()]).T[:,[1,0,2]] - [w/2,h/2,0]
    if len(points)<3:  return None
    pca= TPCA(points)
    normal= pca.EVecs[:,-1]
    #pca2= PCA(n_components=3,svd_solver='full')  #svd_solver='randomized'
    #pca2.fit(points)
    #normal= pca2.components_[-1]
    if pca.EVals[-1]>self.th_normal:  return None
    if normal[2]<0:  normal= -normal
    return normal
  #Get an angle [0,pi] between two features.
  def Diff(self,f1,f2):
    cos_th= np.dot(f1,f2) / (np.linalg.norm(f1)*np.linalg.norm(f2))
    if cos_th>1.0:  cos_th=1.0
    elif cos_th<-1.0:  cos_th=-1.0
    return np.arccos(cos_th)
  #Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  def WSum(self,f1,f2,w1):
    ws= w1*f1 + (1.0-w1)*f2
    ws_norm= np.linalg.norm(ws)
    if ws_norm<1.0e-6:
      raise Exception('TImgPatchFeatNormal: Computing WSum for normals of opposite directions.')
    return ws/ws_norm

#Feature of the average depth of a patch.
class TImgPatchFeatAvrDepth(TImgPatchFeatIF):
  def __init__(self):
    pass
  #Get an average depth of an image patch img_patch.
  def __call__(self,img_patch):
    img_valid= img_patch[img_patch!=0]
    if len(img_valid)==0:    return None
    return np.array([np.mean(img_valid)])
  '''
  #Equal to the above __call__ (for test).
  def __call__(self,img_patch):
    h,w= img_patch.shape[:2]
    #points= [[x-w/2,y-h/2,img_patch[y,x]] for y in range(h) for x in range(w) if img_patch[y,x]!=0]
    points= np.vstack([np.where(img_patch!=0), img_patch[img_patch!=0].ravel()]).T[:,[1,0,2]] - [w/2,h/2,0]
    if len(points)==0:  return None
    return np.array([np.mean(points,axis=0)[-1]])
  '''
  #Get a difference (scholar value) between two features.
  def Diff(self,f1,f2):
    return np.linalg.norm(f1-f2)
  #Compute a weighted sum of two features. i.e. w1*f1 + (1.0-w1)*f2
  def WSum(self,f1,f2,w1):
    return w1*f1 + (1.0-w1)*f2

#Segment img by a given feature model such as normal.
def ClusteringByFeatures(img, w_patch, f_feat=TImgPatchFeatAvrDepth(), th_feat=15):
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
  debug_t_start= time.time()
  nodes= [TNode([(u*w_patch,v*w_patch,w_patch,w_patch)],f_feat(img[v*w_patch:v*w_patch+w_patch,u*w_patch:u*w_patch+w_patch]))
          for v in range(Nv)
          for u in range(Nu)]
  print 'DEBUG:feat cmp time:',time.time()-debug_t_start
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
      if f_feat.Diff(node.feat,node2.feat) < th_feat:
        #print '--merged',debug_id2idx[node2],'-->',debug_id2idx[node]
        #Merge node2 into node:
        r1= float(len(node.patches))/(len(node.patches)+len(node2.patches))
        node.feat= f_feat.WSum(node.feat, node2.feat, r1)
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


#Convert patch points to a binary image.
def PatchPointsToImg(patches):
  if len(patches)==0:  return None,None,None,None
  _,_,patch_w,patch_h= patches[0]
  patches_pts= np.array([[y/patch_h,x/patch_w] for x,y,_,_ in patches])
  patches_topleft= np.min(patches_pts,axis=0)
  patches_btmright= np.max(patches_pts,axis=0)
  patches_img= np.zeros(patches_btmright-patches_topleft+[1,1])
  patches_pts_o= patches_pts - patches_topleft
  patches_img[patches_pts_o[:,0],patches_pts_o[:,1]]= 1
  return patches_img,patches_topleft[::-1]*[patch_w,patch_h],patch_w,patch_h


from binary_seg2 import FindSegments
def DrawClusters(img, clusters):
  img_viz= cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  col_set= ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255))
  for i,node in enumerate(clusters):
    print i, node.feat, 'patches:',len(node.patches),
    col= col_set[i%len(col_set)]
    for patch in node.patches:
      x,y,w,h= patch
      #cv2.circle(img_viz, (x+w/2,y+h/2), (w+h)/4, col, 1)
      cv2.rectangle(img_viz, (x+1,y+1), (x+w-1,y+h-1), np.array(col)/2, 1)
    patches_img,patches_topleft,patch_w,patch_h= PatchPointsToImg(node.patches)
    #print '  debug:',patches_img.shape
    #if patches_img.size>100:
      #cv2.imwrite('patches_img-{0}.png'.format(i), patches_img)
      #cv2.imshow('patches_img-{0}'.format(i),cv2.resize(patches_img,(patches_img.shape[1]*10,patches_img.shape[0]*10),interpolation=cv2.INTER_NEAREST ))
    segments,num_segments= FindSegments(patches_img)
    print 'seg:',[np.sum(segments==idx) for idx in range(1,num_segments+1)]
    for idx in range(1,num_segments+1):
      patches= [patches_topleft + [u*patch_w,v*patch_h] for v,u in zip(*np.where(segments==idx))]
      for patch in patches:
        x,y= patch
        for j in range(0,idx):
          cv2.circle(img_viz, (x+patch_w/2,y+patch_h/2), max(1,(patch_w+patch_h)/4-2*j), col, 1)
  return img_viz


if __name__=='__main__':
  #img_depth= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth']
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth1.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth2.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth3.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth001.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth002.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth003.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth004.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  print img_depth.shape, img_depth.dtype, [np.min(img_depth), np.max(img_depth)]

  t_start= time.time()
  #clusters= ClusteringByFeatures(img_depth, w_patch=25, f_feat=TImgPatchFeatNormal(0.4), th_feat=0.2)
  clusters= ClusteringByFeatures(img_depth, w_patch=25, f_feat=TImgPatchFeatNormal(5.0), th_feat=0.5)
  #clusters= ClusteringByFeatures(img_depth, w_patch=25, f_feat=TImgPatchFeatAvrDepth(), th_feat=3.0)
  clusters= [node for node in clusters if len(node.patches)>=3]
  print 'Number of clusters:',len(clusters)
  print 'Sum of numbers of patches:',sum([len(node.patches) for node in clusters])
  print 'Computation time:',time.time()-t_start
  img_viz= DrawClusters(img_depth, clusters)

  cv2.imshow('depth',img_viz)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

