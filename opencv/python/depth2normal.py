#!/usr/bin/python
#\file    depth2normal.py
#\brief   Convert a depth image to a normal image.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.07, 2021
'''
Refs.
https://github.com/akihikoy/ay_3dvision/blob/master/ay_3dvision/src/pcl_util.cpp#L185
ConvertPointCloudToNormalImage
https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc
'''
import numpy as np
import six.moves.cPickle as pickle
import cv2
import time

def DepthToNormalImg(img_depth, with_amp=False, amp_beta=5.0):
  m= img_depth.astype('int16')
  nx= np.pad((m[:,2:]-m[:,:-2])/2.0, ((0,0),(1,1)), 'constant')
  ny= np.pad((m[2:,:]-m[:-2,:])/2.0, ((1,1),(0,0)), 'constant')
  nz= np.zeros_like(nx)
  img_norm= np.stack((ny,nx,nz)).transpose((1,2,0))
  if with_amp:
    img_amp= np.linalg.norm(img_norm,axis=2)
    img_amp= (np.arctan(img_amp/amp_beta))*512./np.pi
    return img_norm, img_amp
  return img_norm


if __name__=='__main__':
  img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/ongrdepth001.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth'].reshape((480,-1))
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth1.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth2.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/test_depth3.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth001.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth002.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth003.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  #img_depth= cv2.cvtColor(cv2.imread('../cpp/sample/nprdepth004.png'), cv2.COLOR_BGR2GRAY).astype(np.uint16)
  print img_depth.shape, img_depth.dtype, [np.min(img_depth), np.max(img_depth)]

  t_start= time.time()
  img_norm,img_amp= DepthToNormalImg(img_depth, with_amp=True)
  print 'Computation time:',time.time()-t_start

  print np.min(img_norm.reshape(-1,3),axis=0),np.max(img_norm.reshape(-1,3),axis=0)
  print np.min(img_amp),np.max(img_amp)

  cv2.imshow('depth',cv2.cvtColor(img_depth.astype('uint8'), cv2.COLOR_GRAY2BGR))
  cv2.imshow('normal(abs)',np.abs(img_norm))
  cv2.imshow('normal(amp)',img_amp.astype('uint8'))
  img_amp= cv2.GaussianBlur(img_amp,(5,5),0)
  cv2.imshow('normal(amp-blur)',img_amp.astype('uint8'))
  while cv2.waitKey() not in map(ord,[' ','q']):  pass
