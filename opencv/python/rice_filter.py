#!/usr/bin/python3
#\file    rice_filter.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.30, 2025

import cv2
import numpy as np

if __name__=='__main__':
  import sys
  src_img= sys.argv[1] if len(sys.argv)>1 else '../cpp/sample/banana-peel_620x350_71497523358.jpg'

  img= cv2.imread(src_img)
  gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  sobel_x= cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y= cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  edge= cv2.magnitude(sobel_x, sobel_y)
  edge= np.uint8(np.clip(edge, 0, 255))

  _, edge_binary = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)

  kernel_noise= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  edge_binary= cv2.morphologyEx(edge_binary, cv2.MORPH_OPEN, kernel_noise)

  kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  rice_mask= cv2.dilate(edge_binary, kernel)

  add_rgb= np.array([-60, -60, -20], dtype=np.int16)
  result_img= img.astype(np.int16)
  result_img[rice_mask > 0] += add_rgb
  result_img= np.clip(result_img, 0, 255).astype(np.uint8)

  cv2.imshow('Original', img)
  cv2.imshow('Rice Enhanced', result_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

