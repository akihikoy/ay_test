#!/usr/bin/python
#\file    create_depth1.py
#\brief   Generate a depth image for test.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.25, 2020
import numpy as np
import cv2

#NOTE: x, mu, invSig should be np.array
def GaussianN(x, mu, invSig):
  ex= np.exp(-0.5*(x-mu).T.dot(invSig).dot(x-mu))
  return ex

#Feature vector with Gaussian basis function
#NOTE: x, mu, invSig should be np.array
def FeaturesG(x, mu_list, invSig_list):
  return [GaussianN(x,mu,invSig) for mu,invSig in zip(mu_list,invSig_list)]

#NOTE: x, mu, invSig should be np.array
def Quadratic(x, mu, invSig):
  return 0.5*(x-mu).T.dot(invSig).dot(x-mu)

#Feature vector with normalized Gaussian basis function
#NOTE: x, mu, invSig should be np.array
def FeaturesNG(x, mu_list, invSig_list):
  quad= [Quadratic(x,mu,invSig) for mu,invSig in zip(mu_list,invSig_list)]
  quad_max= max(quad)
  gaussian= [np.exp(quad_max-q) for q in quad]
  sum_g= sum(gaussian)  #Should be greater than 1.0
  return [g/sum_g for g in gaussian]

if __name__=='__main__':
  width,height= 640,480
  #img= np.zeros([height,width,1],dtype=np.uint8)
  mu_list= np.array([[100,100], [320,400], [500,150]])
  invSig_list= [np.array([[1.0/150.0,0.0],[0.0,1.0/100.0]])**2,
                np.array([[1.0/100.0,0.0],[0.0,1.0/150.0]])**2,
                np.array([[1.0/150.0,0.0],[0.0,1.0/120.0]])**2]
  w= [60, 80, 50]

  Features= FeaturesG
  #Features= FeaturesNG
  img= np.array([[[np.dot(w,Features([x,y], mu_list, invSig_list))] for x in range(width)] for y in range(height)])
  img= 200.0-img
  #img= img*(255.0/np.max(img))
  img= img.astype(np.uint8)

  filename= '../cpp/sample/test_depth1.png'
  cv2.imwrite(filename,img)
  print 'File is saved into:',filename

  cv2.imshow('depth',img)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass
