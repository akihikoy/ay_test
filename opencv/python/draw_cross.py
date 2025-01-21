#!/usr/bin/python3
#\file    draw_cross.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.11, 2025
import numpy as np
import cv2

'''
Draw a cross mark on the image center.
  img: Input image.
  size: Cross mark pixel size.
  col: Color.
  thickness: Thickness of the lines.
'''
def DrawCrossOnCenter(img, size=20, col=(255,0,128), thickness=1):
  rows,cols,_= img.shape
  hsize= size//2
  cv2.line(img, (cols//2-hsize,rows//2), (cols//2+hsize,rows//2), col, thickness)
  cv2.line(img, (cols//2,rows//2-hsize), (cols//2,rows//2+hsize), col, thickness)


if __name__=='__main__':
  from draw_squares1 import GenSquarePattern1
  img= GenSquarePattern1(300,200,N=50,with_reverse=False)
  DrawCrossOnCenter(img)
  cv2.imshow('image',img)
  while cv2.waitKey(100)&0xFF!=ord('q'):
    pass
  cv2.destroyAllWindows()

