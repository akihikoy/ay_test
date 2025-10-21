#!/usr/bin/python3
#\file    polygon_bb1.2.py
#\brief   polygon minimum area bounding box
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.21, 2025
from polygon_bb1 import *

if __name__=='__main__':
  # Data selection
  from gen_data import *
  choice= input("Select dataset (1:Gen3d_01, 2:Gen3d_02, 3:Gen3d_11, 4:Gen3d_12, 5:Gen3d_13, 6:Gen2d_01 [default]): ").strip()
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
  else:
    points= Gen2d_01()

  bb= TPolygonBoundBox2D(points)

  print('Convex hull points: \n', bb.HullPoints, "\n")
  print("Minimum area bounding box:")
  print("  Rotation angle:", bb.Angle, "rad  (", bb.Angle*(180/math.pi), "deg )")
  print("  Width:", bb.Width, " Height:", bb.Height, "  Area:", bb.Area)
  print("  Center point: \n", bb.Center)
  print("  Corner points: \n", bb.CornerPoints)

  #Plot with matplotlib
  import matplotlib.pyplot as plt

  # Convert to arrays for easy slicing
  orig= array(points)
  hull= array(bb.HullPoints)
  corners= array(bb.CornerPoints)
  center= array(bb.Center)

  # Close the loops for orig, hull and corners
  orig_closed= vstack([orig, orig[:1]])
  hull_closed= vstack([hull, hull[:1]])
  corners_closed= vstack([corners, corners[:1]])

  # Plot original points
  plt.plot(orig_closed[:,0], orig_closed[:,1], '--', linewidth=3.0, label='original', zorder=1)

  # Plot convex hull
  plt.plot(hull_closed[:,0], hull_closed[:,1], ':', linewidth=1.5, label='convex hull', zorder=2)

  # Plot minimum area bounding box corners
  plt.plot(corners_closed[:,0], corners_closed[:,1], '-.', linewidth=1.2, label='min-area box', zorder=3)

  # Plot center point
  plt.plot(center[0], center[1], 'o', markersize=6, label='center', zorder=4)

  # Keep aspect ratio 1:1 like gnuplot's set size ratio -1
  ax= plt.gca()
  ax.set_aspect('equal', adjustable='box')

  plt.title('Minimum Area Bounding Box')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  plt.show()
