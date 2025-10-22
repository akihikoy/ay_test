#!/usr/bin/python3
#\file    polygon_slice2.py
#\brief   Slice a polygon.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.22, 2025
import numpy as np

# Get rotation matrices for angle_rad.
def RotXY(angle_rad):
  c, s = np.cos(angle_rad), np.sin(angle_rad)
  R = np.array([[c, -s],
                [s,  c]])
  return R


"""
Slice a polygon for scanlines (x_range, step) along the x-axis.
For each scanline, only the outer point pairs are sampled.
Notes:
  - Works for convex polygons and returns the "silhouette span".
  - For concave polygons, min/max will bridge across concavities (may include outside area).
"""
def SlicePolygon(contour_xy, step, x_range=None,
                 eps=1e-12, neighbor_eps=1e-9):
  P = np.asarray(contour_xy, dtype=float)
  if P.shape[0] < 3:
    raise ValueError("contour must have at least 3 points")

  # Range of x' to scan
  xmin, xmax = [None, None] if x_range is None else x_range
  if xmin is None:  xmin = np.min(P[:,0])
  if xmax is None:  xmax = np.max(P[:,0])
  if step <= 0:
    raise ValueError("step must be positive")
  print(f'x_range={xmin}, {xmax}, step={step}')

  # Build scan x' coordinates (inclusive with tolerance)
  K = int(np.floor((xmax - xmin) / step + eps)) + 1
  s_coords = xmin + np.arange(K) * step

  # Buckets of intersections per scanline: store (x', y')
  inters = [[] for _ in range(K)]

  N = P.shape[0]
  for i in range(N):
    x0, y0 = P[i]
    x1, y1 = P[(i+1) % N]
    dx = x1 - x0
    dy = y1 - y0

    # Skip edges parallel to scanline x'=const (avoid interval-overlap ambiguity)
    if abs(dx) < eps:
      continue

    # Edge contributes to scanlines s with min(x0,x1) ≤ s < max(x0,x1)
    ex_min, ex_max = (x0, x1) if x0 <= x1 else (x1, x0)
    k0 = int(np.ceil((ex_min - xmin) / step - eps))
    k1 = int(np.floor((ex_max - xmin) / step - eps))  # upper end is open to avoid double count
    if k1 < 0 or k0 >= K:
      continue
    k0 = max(k0, 0)
    k1 = min(k1, K-1)

    for k in range(k0, k1+1):
      sx = s_coords[k]
      # Parameter t along the edge
      t = (sx - x0) / dx
      if t < -eps or t > 1+eps:
        continue
      y = y0 + t * dy
      inters[k].append((sx, y))

  # For each scanline, sort by y' and take only (min, max)
  pts1, pts2= [],[]  # Edge points on the scanlines
  for k in range(K):
    pts_k = inters[k]
    if not pts_k or len(pts_k) < 2:
      # Not enough intersections to form a pair
      continue

    A = np.array(pts_k, dtype=float)  # (M,2): (x', y')
    ys = A[:,1]
    Pmin = A[np.argmin(ys)]
    Pmax = A[np.argmax(ys)]

    # Ignore too-close intersections
    if abs(Pmax[1] - Pmin[1]) < neighbor_eps:
      continue

    pts1.append(Pmin)
    pts2.append(Pmax)

  return np.array(pts1, dtype=float), np.array(pts2, dtype=float)


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

  # BB fitting method
  #choice_bb= input("Select the BB fit method (1: MinAreaRect, 2: BoundingBoxWithEllipseAxis [default]): ").strip()

  for choice_bb in ('1', '2'):

    if choice_bb=='1':
      from polygon_min_area_rect import MinAreaRect
      center,size,angle= MinAreaRect(points)
      title = 'SlicePolygon (MinAreaRect)'
    else:
      from polygon_bb4 import BoundingBoxWithEllipseAxis
      center,size,angle= BoundingBoxWithEllipseAxis(points)
      title = 'SlicePolygon (BoundingBoxWithEllipseAxis)'

    rot= np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rect= [np.array(center)+rot.dot(p) for p in [[size[0]*0.5,size[1]*0.5],[size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,size[1]*0.5]]]

    #xy_samples = SlicePolygon(points, -angle, step=0.05, x_range=None)
    #xy_samples = SlicePolygon(points, -angle+np.pi/2., step=0.05, x_range=[-0.4, 0.1])

    # Range in percent (1=100%)
    #s_range_p = [0., 1.]
    s_range_p = [0.1, 0.5]
    R = RotXY(-angle)
    # Rotate all points
    points_rot = (R @ np.asarray(points, dtype=float).T).T  # (N,2)
    s_range = [np.min(points_rot[:,0]), np.max(points_rot[:,0])]
    s_len = s_range[1]-s_range[0]
    step = s_len / 50
    s_range = [s_range[0]+s_len*s_range_p[0], s_range[0]+s_len*s_range_p[1]]
    pts1_rot, pts2_rot = SlicePolygon(points_rot, step, x_range=s_range)
    # Map back to original XY
    pts1, pts2 = (R.T @ pts1_rot.T).T, (R.T @ pts2_rot.T).T


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

    # Overlay sampled pairs
    if len(pts1) > 0 and len(pts2) > 0:
      plotted_label = False  # show legend only once
      for p1, p2 in zip(pts1, pts2):
        if not plotted_label:
          plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', linewidth=1.0, label='xy_samples', zorder=2)
          plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o', markersize=2, zorder=2, label='_nolegend_')
          plotted_label = True
        else:
          plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', linewidth=1.0, zorder=2, label='_nolegend_')
          plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o', markersize=2, zorder=2, label='_nolegend_')

    # Keep aspect ratio 1:1 like gnuplot's set size ratio -1
    ax= plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

