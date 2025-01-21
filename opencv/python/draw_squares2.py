#!/usr/bin/python3
#\file    draw_squares2.py
#\brief   Draw squares (no overlap).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.22, 2021
import numpy as np
import cv2

def GenSquarePattern2(img_w=200, img_h=200, w=20, h=20, N=10,
                      bg_col=(255,255,255), line_col=(2,48,155), fill_col=(86,168,228)):
  Nw,Nh= img_w//w,img_h//h
  i_pts= np.random.permutation(Nw*Nh)[:N]
  pts= [(i//Nw,i%Nw) for i in i_pts]

  img= np.ones((img_h,img_w,3), np.uint8)*np.array(bg_col, np.uint8)
  for u,v in pts:
    cv2.rectangle(img, (u*w,v*h), (u*w+w-1,v*h+h-1), fill_col, -1)
    cv2.rectangle(img, (u*w,v*h), (u*w+w-1,v*h+h-1), line_col, 2)
  return img

if __name__=='__main__':
  img= GenSquarePattern2(N=50)
  cv2.imshow('image',img)
  while cv2.waitKey(100)&0xFF!=ord('q'):
    pass
  cv2.destroyAllWindows()
