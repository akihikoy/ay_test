#!/usr/bin/python3
#\file    polygon_slice2_test.py
#\brief   Test code of polygon_bb2.cpp
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.22, 2025
import subprocess, sys, json, numpy as np

def run_slice_cpp(points, exe='./polygon_slice2_test.out', choice_bb='1', s_range_p=[0., 1.]):
  # Prepare stdin payload
  payload = "\n".join(
    " ".join(str(float(x)) for x in p[:3]) for p in points
  ).encode('utf-8')
  args = [exe, choice_bb, *map(str, s_range_p)]
  proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = proc.communicate(payload)
  if proc.returncode != 0:
    sys.stderr.write(err.decode())
    raise RuntimeError(f"{exe} failed")
  sys.stderr.write(err.decode())

  res = json.loads(out.decode('utf-8'))
  center = tuple(res["center"])
  size   = tuple(res["size"])
  angle  = float(res["angle"])
  title  = res.get("title", "SlicePolygon")
  pts1   = np.array(res.get("pts1", []), dtype=float)
  pts2   = np.array(res.get("pts2", []), dtype=float)
  return center, size, angle, pts1, pts2, title

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

  for choice_bb in ('1', '2'):

    s_range_p = [0., 1.]  # percent range on the rotated x
    s_range_p = [0.1, 0.5]
    center, size, angle, pts1, pts2, title = run_slice_cpp(points, exe='./polygon_slice2_test.out', choice_bb=choice_bb, s_range_p=s_range_p)

    rot= np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rect= [np.array(center)+rot.dot(p) for p in [[size[0]*0.5,size[1]*0.5],[size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,-size[1]*0.5],[-size[0]*0.5,size[1]*0.5]]]

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


