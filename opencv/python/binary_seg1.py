#!/usr/bin/python
#\file    binary_seg1.py
#\brief   Segmentation of small binary image;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.27, 2020
import numpy as np
import cv2

if __name__=='__main__':
  img = cv2.imread('../cpp/sample/binary2.png')
  print img.shape

  disp_int32 = lambda img: np.uint8((img*2**8+(img>0)*2**10)/2**4)
  def disp_img(win_name, img, resize=(20,20)):
    cv2.imshow(win_name, cv2.resize(img,(img.shape[1]*resize[0],img.shape[0]*resize[1]),interpolation=cv2.INTER_NEAREST ))
    while cv2.waitKey() not in map(ord,[' ','q']):  pass

  disp_img('input', img*255)

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  #disp_img('thresh',thresh)

  ## noise removal
  #kernel = np.ones((3,3),np.uint8)
  #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
  #disp_img('opening',opening)
  opening = gray

  # sure background area
  sure_bg = cv2.dilate(opening,np.ones((1,1),np.uint8),iterations=3)
  print 'sure_bg\n',sure_bg
  disp_img('sure_bg',disp_int32(sure_bg))

  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(gray,cv2.cv.CV_DIST_L2,5)
  disp_img('dist_transform',dist_transform/10)
  ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
  print 'sure_fg\n',np.uint8(sure_fg/255)
  disp_img('sure_fg',sure_fg)

  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)


  ## Marker labelling
  #ret, markers = cv2.connectedComponents(sure_fg)
  ## Add one to all labels so that sure background is not 0, but 1
  #markers = markers+1
  cnts,_= cv2.findContours(sure_fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  print 'cnts\n',cnts
  markers = np.zeros(sure_fg.shape,np.int32)
  num_markers = len(cnts)
  for idx,cnt in enumerate(cnts):
    cv2.drawContours(markers, [cnt], -1, idx+1, 2, 8)
  markers = markers + 1
  print 'markers(contours)\n',markers
  disp_img('markers(contours)',disp_int32(markers))


  # Now, mark the region of unknown with zero
  markers[unknown>0] = 0
  print 'markers(removing unknown)\n',markers
  disp_img('markers(removing unknown)',disp_int32(markers))

  cv2.watershed(img,markers)
  print 'markers(watershed)\n',markers
  disp_img('markers(watershed)',disp_int32(markers))

  img[markers == -1] = np.array([1,0,0],np.uint8)
  for idx in range(2,num_markers+2):
    img[markers == idx] = np.array([0,idx*2,0],np.uint8)

  disp_img('image',disp_int32(img))
