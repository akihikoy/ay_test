#!/usr/bin/python3
#\file    cam_mat.py
#\brief   Camera matrix from camera calibration parameters;
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.22, 2020
import cv2
import numpy as np

#Camera parameters (ELP) obtained by camera calibration.
K= np.array([ 2.9626599933569219e+02, 0., 1.6184603192794432e+02, 0.,
      2.9626599933569219e+02, 1.2062472517136455e+02, 0., 0., 1. ]).reshape(3,3)
D= np.array([ -9.5718286674593955e-01, 5.5100676290063433e-01, 0., 0., 0. ]).reshape(1,5)
R= np.array([ 1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0 ]).reshape(3,3)


alpha= 0.8
size_in= (320,240)
size_out= (320,240)
P,roi= cv2.getOptimalNewCameraMatrix(K, D, size_in, alpha, size_out)

print('''
K= {K}
D= {D}
alpha= {alpha}
size_in,size_out= {size_in}, {size_out}
P= {P}
roi= {roi}
'''.format(K=K,D=D,alpha=alpha,size_in=size_in,size_out=size_out,P=P,roi=roi))
