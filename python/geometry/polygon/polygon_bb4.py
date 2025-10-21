#!/usr/bin/python3
#\file    polygon_bb4.py
#\brief   Polygon bounding box detection with ellipse axis estimation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.21, 2025
import numpy as np
#from polygon_com_2d import PolygonCentroid2D
from scipy.spatial import ConvexHull

# Oriented bounding box whose long edge is parallel to the ellipse major axis.
#   Expect XY: (N,2) float-like.
def BoundingBoxWithEllipseAxis(XY, trim_p=0.0, ridge=1e-3):
  XY = np.asarray(XY, dtype=float)
  n = XY.shape[0]
  if n == 0:
    raise ValueError("Empty input")
  if n == 1:
    cx, cy = float(XY[0,0]), float(XY[0,1])
    return (cx, cy), (0.0, 0.0), 0.0

  # -- 1) Center (use simple mean for speed; polygon centroid is heavier) --
  centroid = XY.mean(axis=0)
  dx = (XY[:,0] - centroid[0]).reshape(-1, 1)
  dy = (XY[:,1] - centroid[1]).reshape(-1, 1)

  # -- 2) ef6-like linear system with ridge to get conic params (A,B,C,D,E; F=-1) --
  # Design: [xx, xy, yy, x, y] * p ≈ 1
  A = np.hstack([dx*dx, dx*dy, dy*dy, dx, dy])   # (N,5)
  b = np.ones((n, 1))                            # (N,1)
  AtA = A.T @ A + ridge * np.eye(5)
  Atb = A.T @ b
  p = np.linalg.solve(AtA, Atb).ravel()          # [A,B,C,D,E]
  Acoef, Bcoef, Ccoef = p[0], p[1], p[2]

  # -- 3) Major-axis angle from conic (ef6's ellipse_angle_of_rotation) --
  b_half = 0.5 * Bcoef
  if b_half == 0.0:
    angle = 0.0 if Acoef > Ccoef else np.pi * 0.5
  else:
    denom = (Acoef - Ccoef)
    if Acoef > Ccoef:
      angle = 0.5 * np.arctan(2.0 * b_half / denom)
    else:
      angle = np.pi * 0.5 + 0.5 * np.arctan(2.0 * b_half / denom)
  # Normalize to [-pi/2, pi/2)
  angle = (angle + np.pi/2.0) % np.pi - np.pi/2.0

  # -- 5) Project points to the (angle)-aligned axes (no rotation matrix) --
  ca, sa = np.cos(angle), np.sin(angle)
  pdx = XY[:,0] - centroid[0]
  pdy = XY[:,1] - centroid[1]
  proj_x = pdx * ca + pdy * sa        # along major axis
  proj_y = -pdx * sa + pdy * ca       # along minor axis

  # -- 6) Percentile trimming (optional) --
  if trim_p > 0.0:
    x_lo = np.percentile(proj_x, trim_p)
    x_hi = np.percentile(proj_x, 100.0 - trim_p)
    y_lo = np.percentile(proj_y, trim_p)
    y_hi = np.percentile(proj_y, 100.0 - trim_p)
  else:
    x_lo, x_hi = proj_x.min(), proj_x.max()
    y_lo, y_hi = proj_y.min(), proj_y.max()

  w = float(x_hi - x_lo)
  h = float(y_hi - y_lo)

  # -- 7) BBox center in world coords: midpoint in rotated frame, then unrotate + shift --
  mid_x = 0.5 * (x_lo + x_hi)
  mid_y = 0.5 * (y_lo + y_hi)
  cx_w = float(centroid[0] + mid_x * ca - mid_y * sa)
  cy_w = float(centroid[1] + mid_x * sa + mid_y * ca)

  return (cx_w, cy_w), (w, h), float(angle)


if __name__=='__main__':
  # Data selection
  from gen_data import *
  choice= input("Select dataset (1:Gen3d_01, 2:Gen3d_02, 3:Gen3d_11, 4:Gen3d_12, 5:Gen3d_13, 6:Gen3d_14, 7:Gen3d_14(dense=1), 8:Gen3d_15, 9:Gen3d_15(dense=1), a:Gen2d_01 [default]): ").strip()
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
  elif choice == "8":
    points= To2d2(Gen3d_15())
  elif choice == "9":
    points= To2d2(Gen3d_15(dense=1.0))
  else:
    points= Gen2d_01()
  #print(points)


  center,size,angle= BoundingBoxWithEllipseAxis(points)

  rot= np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  rect= [np.array(center)+rot.dot(p) for p in [[size[0]*0.5,size[1]*0.5],[size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,size[1]*0.5]]]

  print(f"BoundingBoxWithEllipseAxis:")
  print("  Rotation angle:", angle, "rad  (", angle*(180/math.pi), "deg )")
  print("  Width:", size[0], " Height:", size[1], "  Area:", size[0]*size[1])
  print("  Center point: \n", center)
  print("  Corner points: \n", rect)

  #Plot with matplotlib
  import matplotlib.pyplot as plt

  # Convert to arrays for easy slicing
  orig= np.array(points)
  corners= np.array(rect)
  center= np.array(center)

  # Close the loops for orig and corners
  orig_closed= np.vstack([orig, orig[:1]])
  corners_closed= np.vstack([corners, corners[:1]])

  # Plot original points
  plt.plot(orig_closed[:,0], orig_closed[:,1], '--', linewidth=3.0, label='original', zorder=1)

  # Plot bounding box corners
  plt.plot(corners_closed[:,0], corners_closed[:,1], '-.', linewidth=1.2, label='bounding box', zorder=3)

  # Plot center point
  plt.plot(center[0], center[1], 'o', markersize=6, label='center', zorder=4)

  # Keep aspect ratio 1:1 like gnuplot's set size ratio -1
  ax= plt.gca()
  ax.set_aspect('equal', adjustable='box')

  plt.title(f'BoundingBoxWithEllipseAxis')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  plt.show()

