#!/usr/bin/python
#\file    localmax2.py
#\brief   Find local maxima on 2d depth image.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.25, 2020
import numpy as np
import six.moves.cPickle as pickle
import cv2
import scipy.ndimage


#Based on the answer in: https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detect_peaks1(image):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """
  #image= scipy.ndimage.gaussian_filter(image, sigma=5)
  image= image.reshape(480,640)

  # define an 8-connected neighborhood
  neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)
  neighborhood_size = 2

  #apply the local maximum filter; all pixel of maximal value
  #in their neighborhood are set to 1
  #local_max = scipy.ndimage.filters.maximum_filter(image, footprint=neighborhood)==image
  #local_max = scipy.ndimage.filters.minimum_filter(image, footprint=neighborhood)==image
  local_max = scipy.ndimage.filters.maximum_filter(image, neighborhood_size)==image
  #local_max = scipy.ndimage.filters.minimum_filter(image, neighborhood_size)==image
  #local_max is a mask that contains the peaks we are
  #looking for, but also the background.
  #In order to isolate the peaks we must remove the background from the mask.
  print np.count_nonzero(local_max==False),np.count_nonzero(local_max)
  #cv2.imshow('depth2',local_max)

  #we create the mask of the background
  background = (image>0)

  #a little technicality: we must erode the background in order to
  #successfully subtract it form local_max, otherwise a line will
  #appear along the background border (artifact of the local maximum filter)
  eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

  #we obtain the final mask, containing only peaks,
  #by removing the background from the local_max mask (xor operation)
  detected_peaks = local_max ^ eroded_background

  #return detected_peaks
  #print detected_peaks, detected_peaks.shape
  #print np.where(detected_peaks==True)
  #print np.count_nonzero(detected_peaks==False)+np.count_nonzero(detected_peaks)
  rows,cols= np.where(detected_peaks)
  return zip(cols,rows)


#Based on the answer in: https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
def detect_peaks2(image):
  #image= scipy.ndimage.gaussian_filter(image, sigma=2)
  image= image.reshape(480,640)
  neighborhood_size = 7
  threshold = 300

  data_max = scipy.ndimage.filters.maximum_filter(image, neighborhood_size)
  maxima = (image == data_max)
  data_min = scipy.ndimage.filters.minimum_filter(image, neighborhood_size)
  diff = ((data_max - data_min) > threshold)
  maxima[diff == 0] = 0

  labeled, num_objects = scipy.ndimage.label(maxima)
  slices = scipy.ndimage.find_objects(labeled)
  detected_peaks= []
  #print len(slices)
  for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    y_center = (dy.start + dy.stop - 1)/2
    detected_peaks.append([x_center,y_center])
  return detected_peaks


if __name__=='__main__':
  img_depth= pickle.load(open('../../python/data/depth001.dat','rb'))['img_depth']
  #img_depth= cv2.cvtColor(cv2.imread('/tmp/obs_img_depth.png'), cv2.COLOR_BGR2GRAY)
  #print img_depth.shape

  localmax= detect_peaks1(img_depth)
  #localmax= detect_peaks2(img_depth)
  #print localmax
  print len(localmax)

  for u,v in localmax:
    cv2.circle(img_depth, (u,v), 2, 255, 1)

  cv2.imshow('depth',img_depth*150)
  #cv2.imshow('localmax',localmax)
  while cv2.waitKey() not in map(ord,[' ','q']):  pass

