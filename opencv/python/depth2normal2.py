#!/usr/bin/python
#\file    depth2normal2.py
#\brief   Convert a depth image to a normal image with considering 3D geometry.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.08, 2021
import numpy as np
import six.moves.cPickle as pickle
import cv2
import time

def DepthToNormalCore(img_depth, proj_mat):
  m= img_depth.astype('int16')
  Fx,Fy,Cx,Cy= proj_mat[0,0],proj_mat[1,1],proj_mat[0,2],proj_mat[1,2]
  Fz= 1e3
  u= np.repeat([range(m.shape[1])],m.shape[0],axis=0)
  v= np.repeat([range(m.shape[0])],m.shape[1],axis=0).T
  d1= np.pad(m[:,1:], ((0,0),(0,1)), 'constant')
  d3= np.pad(m[:,:-1], ((0,0),(1,0)), 'constant')
  d2= np.pad(m[1:,:], ((0,1),(0,0)), 'constant')
  d4= np.pad(m[:-1,:], ((1,0),(0,0)), 'constant')
  ny= (d1+d3)*(d2-d4)/(Fx*Fz*Fz)
  nx= (d1-d3)*(d2+d4)/(Fy*Fz*Fz)
  nz= ((u+v+(-Cx-Cy+1))*d1*d2 + (-u+v+(+Cx-Cy+1))*d2*d3 + (-u-v+(Cx+Cy+1))*d3*d4 + (u-v+(-Cx+Cy+1))*d1*d4)/(Fx*Fy*Fz*Fz)
  alpha= np.arctan2(ny, nx)  #Angle of the normal projected on xy plane from x axis
  beta= np.abs(np.arctan2(nz, np.sqrt(nx*nx+ny*ny)))  #Angle of the normal from xy plane
  #print np.min(nz), np.max(nz)
  return alpha,beta

def DepthToNormal(img_depth, proj_mat, resize_ratio=0.5, ks_gauss=5):
  if resize_ratio==1 or resize_ratio is None:
    img_depth= cv2.GaussianBlur(img_depth,(ks_gauss,ks_gauss),0)
    return DepthToNormalCore(img_depth, proj_mat)
  else:
    img_depth_s= cv2.resize(img_depth, (int(img_depth.shape[1]*resize_ratio),int(img_depth.shape[0]*resize_ratio)))
    proj_mat_s= GetProjMatForResizedImg(proj_mat, resize_ratio)
    img_depth_s= cv2.GaussianBlur(img_depth_s,(ks_gauss,ks_gauss),0)
    alpha_s,beta_s= DepthToNormalCore(img_depth_s, proj_mat)
    alpha= cv2.resize(alpha_s, (img_depth.shape[1],img_depth.shape[0]))
    beta= cv2.resize(beta_s, (img_depth.shape[1],img_depth.shape[0]))
    return alpha,beta

#Get a projection matrix for resized image.
def GetProjMatForResizedImg(P, resize_ratio):
  P= resize_ratio*P
  P[2,2]= 1.0
  return P

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

  proj_mat= np.array([[612.449462890625, 0.0, 317.5238952636719, 0.0], [0.0, 611.5702514648438, 237.89498901367188, 0.0], [0.0, 0.0, 1.0, 0.0]])

  t_start= time.time()
  norm_alpha,norm_beta= DepthToNormal(img_depth, proj_mat, resize_ratio=0.25)
  print 'Computation time:',time.time()-t_start

  print [np.min(norm_alpha),np.max(norm_alpha)], [np.min(norm_beta),np.max(norm_beta)]

  beta_img= ((1.0-norm_beta/(0.5*np.pi))*255.).astype('uint8')
  hsvimg= np.dstack(((norm_alpha/np.pi*127.+128.).astype('uint8'),
                     (np.ones_like(norm_alpha)*255).astype('uint8'),
                     beta_img,
                     ))

  cv2.imshow('depth',cv2.cvtColor(img_depth.astype('uint8'), cv2.COLOR_GRAY2BGR))
  cv2.imshow('normal',cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR))
  #beta_img[beta_img<220]= 0
  cv2.imshow('normal(beta)',beta_img)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass
