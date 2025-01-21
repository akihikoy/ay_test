#!/usr/bin/python3
#\file    localmax.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.11, 2019
import numpy as np
import six.moves.cPickle as pickle
import cv2
#import scipy.ndimage

'''
def findIsolatedLocalMaxima(greyScaleImage):
  #Smoothing image.
  greyScaleImage= cv2.bilateralFilter(greyScaleImage,99,75,75)

  squareDiameterLog3 = 3 #27x27

  total = greyScaleImage
  for axis in range(2):
    d = 1
    for i in range(squareDiameterLog3):
      total = np.maximum(total, np.roll(total, d, axis))
      total = np.maximum(total, np.roll(total, -d, axis))
      d *= 3

  maxima = total == greyScaleImage
  h,w = greyScaleImage.shape

  img_maxima= np.zeros((h,w), dtype=np.uint8)
  img_maxima[maxima]= 255

  #result = []
  #for j in range(h):
    #for i in range(w):
      #if maxima[j][i]:
        #result.append((i, j))
  #return result
  ##return img_maxima

  #cnts = cv2.findContours(img_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #cnts = imutils.grab_contours(cnts)

  #result= []
  ## loop over the contours
  #for c in cnts:
    ## compute the center of the contour
    #M = cv2.moments(c)
    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])
    #result.append((cX, cY))
  #return result

  fcres= cv2.findContours(img_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts= fcres[0] if len(fcres)==2 else fcres[1]
  print cnts
  result= []
  # loop over the contours
  for c in cnts:
    cX,cY= np.mean(c,axis=0)[0]
    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])
    result.append((int(cX), int(cY)))
  return result
'''

def FindLocalMaxima(img_depth, ground_depth):
  img_depth= img_depth.copy()
  #Removing ground
  img_depth[img_depth>ground_depth]= 0
  #Converting to uint8
  img_depth= (img_depth).astype('uint8')

  img_depth= cv2.dilate(img_depth,np.ones((5,5),np.uint8),iterations=1)
  img_depth= cv2.erode(img_depth,np.ones((20,20),np.uint8),iterations=1)
  cv2.imshow('depth2',img_depth)

  fcres= cv2.findContours(img_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts= fcres[0] if len(fcres)==2 else fcres[1]
  print(cnts, len(cnts))
  result= []
  for c in cnts:
    cX,cY= np.mean(c,axis=0)[0]
    result.append((int(cX), int(cY)))
  return result


if __name__=='__main__':
  img_depth= pickle.load(open('../../python/data/depth001.dat','rb'), encoding='latin1')['img_depth']
  #img_depth= cv2.cvtColor(cv2.imread('/tmp/obs_img_depth.png'), cv2.COLOR_BGR2GRAY)
  #print img_depth.shape

  #img_depth= scipy.ndimage.gaussian_filter(img_depth, sigma=2)

  #localmax= findIsolatedLocalMaxima(img_depth)
  localmax= FindLocalMaxima(img_depth, 380)
  print(localmax)

  for u,v in localmax:
    cv2.circle(img_depth, (u,v), 4, 255, 1)

  cv2.imshow('depth',img_depth*150)
  #cv2.imshow('localmax',localmax)
  while cv2.waitKey() & 0xFF not in map(ord,[' ','q']):  pass

