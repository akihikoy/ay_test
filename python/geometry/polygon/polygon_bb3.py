#!/usr/bin/python3
#\file    polygon_bb3.py
#\brief   Polygon bounding box detection with Ellipse fitting.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.21, 2025
import numpy as np
from ellipse_fit1 import EllipseFit2D as ef1_EllipseFit2D
from ellipse_fit2 import EllipseFit2D as ef2_EllipseFit2D
from ellipse_fit3 import EllipseFit2D as ef3_EllipseFit2D
from ellipse_fit4 import EllipseFit2D as ef4_EllipseFit2D
from ellipse_fit5 import EllipseFit2D as ef5_EllipseFit2D
from ellipse_fit6 import EllipseFit2D as ef6_EllipseFit2D
from ellipse_fit7 import EllipseFit2D as ef7_EllipseFit2D
from ellipse_fit8 import EllipseFit2D as ef8_EllipseFit2D
from ellipse_fit9 import EllipseFit2D as ef9_EllipseFit2D
from weighted_ellipse_fit2 import SampleWeightedEllipseFit2D


if __name__=='__main__':
  # Data selection
  from gen_data import *
  choice= input("Select dataset (1:Gen3d_01, 2:Gen3d_02, 3:Gen3d_11, 4:Gen3d_12, 5:Gen3d_13, 6:Gen3d_14, 7:Gen3d_14(dense=1.0), 8:Gen2d_01 [default]): ").strip()
  if choice == "1":
    points= To2d2(Gen3d_01())
  elif choice == "2":
    points= To2d2(Gen3d_02())
  elif choice == "3":
    points= To2d2(Gen3d_11())
  elif choice == "4":
    points= To2d2(Gen3d_12())
  elif choice == "5":
    points= To2d2(Gen3d_13())
  elif choice == "6":
    points= To2d2(Gen3d_14())
  elif choice == "7":
    points= To2d2(Gen3d_14(dense=1.0))
  else:
    points= Gen2d_01()
  #print(points)

  # Ellipse fitting method selection
  choice_m= input("Select ellipse fit method (1-9: efX_EllipseFit2D, a: SampleWeighted): ").strip()
  methods= {
    '1': ef1_EllipseFit2D,  #SVD-based ellipse estimation
    '2': ef2_EllipseFit2D,  #Fitzgibbon type direct ellipse fit (algebraic distance minimization with strict ellipse constraint, no regularization)
    '3': ef3_EllipseFit2D,  #Regularized and scaled version of ef2 for improved numerical stability
    '4': ef4_EllipseFit2D,  #Simple linear least squares fit without ellipse constraint
    '5': ef5_EllipseFit2D,  #Bounded linear least squares enforcing semi-positive ellipse parameters
    '6': ef6_EllipseFit2D,  #Ridge-regularized least squares fit with weak stability improvement
    '7': ef7_EllipseFit2D,  #Nonlinear geometric-distance minimization using SVD-based initialization
    '8': ef8_EllipseFit2D,  #SVD-based ellipse estimation using data covariance (center from mean)
    '9': ef9_EllipseFit2D,  #SVD-based ellipse estimation using polygon centroid as center
    'a': SampleWeightedEllipseFit2D,
  }
  if choice_m not in methods:  choice_m= '1'
  method= methods.get(choice_m, ef1_EllipseFit2D)

  if method == SampleWeightedEllipseFit2D:
    c,r1,r2,angle= SampleWeightedEllipseFit2D(points,[1.0]*len(points))
  else:
    c,r1,r2,angle= method(points)

  center,size= c,[r1*2, r2*2]
  rot= np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  rect= [np.array(center)+rot.dot(p) for p in [[size[0]*0.5,size[1]*0.5],[size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,size[1]*0.5]]]

  print(f"Ellipse-fitting ({choice_m}):")
  print("  Rotation angle:", angle, "rad  (", angle*(180/math.pi), "deg )")
  print("  Width:", r1, " Height:", r2, "  Area:", r1*r2)
  print("  Center point: \n", c)
  print("  Corner points: \n", rect)

  #Plot with matplotlib
  import matplotlib.pyplot as plt

  # Convert to arrays for easy slicing
  orig= np.array(points)
  corners= np.array(rect)
  center= np.array(c)

  # Close the loops for orig and corners
  orig_closed= np.vstack([orig, orig[:1]])
  corners_closed= np.vstack([corners, corners[:1]])

  # Plot original points
  plt.plot(orig_closed[:,0], orig_closed[:,1], '--', linewidth=3.0, label='original', zorder=1)

  # Plot ellipse bounding box corners
  plt.plot(corners_closed[:,0], corners_closed[:,1], '-.', linewidth=1.2, label='ellipse box', zorder=3)

  # Plot center point
  plt.plot(center[0], center[1], 'o', markersize=6, label='center', zorder=4)

  # Keep aspect ratio 1:1 like gnuplot's set size ratio -1
  ax= plt.gca()
  ax.set_aspect('equal', adjustable='box')

  plt.title(f'Ellipse-fitting ({choice_m}) Bounding Box')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  plt.show()

