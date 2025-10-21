#!/usr/bin/python3
#\file    polygon_bb_cmp.py
#\brief   Comparison of polygon bounding box detection method.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.21, 2025

from polygon_min_area_rect import MinAreaRect
from polygon_bb4 import BoundingBoxWithEllipseAxis
import matplotlib.pyplot as plt
import time

# --- Helper: build rectangle corner points from (center,size,angle) ---
def rect_from_bbox(center, size, angle):
  # Return 4 corner points (x,y) in CCW order starting from (+w/2,+h/2) in the local frame.
  cx, cy = center
  w, h = size
  ca, sa = np.cos(angle), np.sin(angle)
  # Local corners
  local = np.array([[ +w*0.5, +h*0.5],
                    [ +w*0.5, -h*0.5],
                    [ -w*0.5, -h*0.5],
                    [ -w*0.5, +h*0.5]])
  # Rotate then translate
  rot = np.array([[ ca, -sa],
                  [ sa,  ca]])
  corners = (local @ rot.T) + np.array([cx, cy])
  return corners

def get_bbox_score(area, size, ref_area=None, ref_ratio=None, w_ratio=0.3):
  """
  Compute a heuristic score for a bounding box based on its area and aspect ratio.

  Smaller score = better (tighter and more natural box).

  Parameters
  ----------
  area : float
      Bounding box area.
  size : array-like of shape (2,)
      [width, height].
  ref_area : float or None
      Reference (expected or median) area for normalization.
      If None, only the raw magnitude of area is used.
  ref_ratio : float or None
      Reference aspect ratio (width/height). If None, assumes ratio ≈ 1.
  w_ratio : float
      Weight for aspect-ratio deviation term.

  Returns
  -------
  score : float
      Combined heuristic score (smaller is better).
  """
  w, h = float(size[0]), float(size[1])
  ratio = w / (h + 1e-12)

  # Normalize area (smaller better)
  if ref_area is not None:
    norm_area = area / (ref_area + 1e-12)
  else:
    norm_area = area

  # Deviation from expected ratio (closer to ref_ratio → better)
  if ref_ratio is not None:
    ratio_dev = abs(ratio - ref_ratio) / (ref_ratio + 1e-12)
  else:
    ratio_dev = abs(np.log(ratio))  # penalize deviation from 1.0

  score = norm_area + w_ratio * ratio_dev
  return score

# --- Define algorithms to compare (name, function, kwargs) ---
# NOTE:
# - OrientedBBoxFromEf6Axis: your ef6-angle version (provided earlier in this chat)
# - BoundingBoxWithEllipseAxis: your original SVD-angle version (ef9-angle)
#   If you renamed/deleted it, remove the corresponding entries below.
algs = [
  ("MinAreaRect", MinAreaRect, dict()),
  ("Elp-axis (tr=0.0, rg=1e-3)", BoundingBoxWithEllipseAxis, dict(trim_p=0.0, ridge=1e-3)),
  ("Elp-axis (tr=2.0, rg=1e-3)", BoundingBoxWithEllipseAxis, dict(trim_p=2.0, ridge=1e-3)),
  ("Elp-axis (tr=0.0, rg=1e-2)", BoundingBoxWithEllipseAxis, dict(trim_p=0.0, ridge=1e-2)),
  ("Elp-axis (tr=0.0, rg=1e-6)", BoundingBoxWithEllipseAxis, dict(trim_p=0.0, ridge=1e-6)),
]

# --- Run and plot for multiple algorithms ---
def compare_bboxes(points):
  points = np.asarray(points, dtype=float)

  results = []

  for name, func, kwargs in algs:
    try:
      t0 = time.perf_counter()
      center, size, angle = func(points, **kwargs)
      corners = rect_from_bbox(center, size, angle)
      t1 = time.perf_counter()
      elapsed_ms = (t1 - t0) * 1000.0

      area = float(size[0] * size[1])
      score = get_bbox_score(area, size)

      results.append(dict(
        name=name, center=np.asarray(center), size=np.asarray(size),
        angle=float(angle), corners=np.asarray(corners), area=area,
        elapsed_ms=elapsed_ms, score=score,
      ))

      # Console report per method
      print(f"{name}:")
      print("  Rotation angle:", angle, "rad  (", angle*(180/math.pi), "deg )")
      print("  Width:", size[0], " Height:", size[1], "  Area:", area)
      print("  Center point:", center)
      print("  Corner points:\n", corners)
    except NameError:
      print(f"{name}: function not found; skipped.")
    except Exception as e:
      print(f"{name}: failed with error: {e}")

  # ---- Summary ----
  print("\nSummary:")
  print(f"{'name':30s} {'elapsed [ms]':>12s} {'area':>12s} {'size [w×h]':>20s} {'score':>12s}")
  for r in results:
    name = r['name']
    elapsed = r.get('elapsed_ms', float('nan'))
    area = r.get('area', float('nan'))
    size = r.get('size', [float('nan'), float('nan')])
    score = r.get('score', float('nan'))
    w, h = size[0], size[1]
    print(f"{name:30s} {elapsed:12.3f} {area:12.6g} {w:10.5f}×{h:10.5f} {score:12.6g}")

  # Plot
  fig = plt.figure()
  ax = plt.gca()

  # Original polygon/points (dashed, thicker)
  orig = np.asarray(points)
  if orig.ndim == 2 and orig.shape[0] >= 2:
    orig_closed = np.vstack([orig, orig[:1]])
    ax.plot(orig_closed[:,0], orig_closed[:,1], '--', linewidth=3.0, label='original', zorder=1)
  elif orig.shape[0] == 1:
    ax.plot(orig[0,0], orig[0,1], 'o', label='original', zorder=1)

  # Distinct styles for boxes
  line_styles = ['-', '-.', ':', (0, (5, 2)), (0, (3, 2, 1, 2))]
  widths = [1.6, 1.4, 1.4, 1.2, 1.2]

  for i, r in enumerate(results):
    corners = np.asarray(r['corners'])
    if corners.ndim != 2 or corners.shape[0] < 3:
      print(f"Warning: invalid corners for {r['name']}, skipped.")
      continue
    corners_closed = np.vstack([corners, corners[:1]])

    ls = line_styles[i % len(line_styles)]   # may be tuple; OK when passed as 'linestyle'
    lw = widths[i % len(widths)]

    ax.plot(corners_closed[:,0], corners_closed[:,1],
            linestyle=ls, linewidth=lw, label=r['name'], zorder=3)
    c = np.asarray(r['center'])
    ax.plot(c[0], c[1], marker='o', markersize=5, zorder=4)

  ax.set_aspect('equal', adjustable='box')
  ax.set_title('Bounding Box Comparison')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.grid(True)
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
  plt.tight_layout()
  plt.show()


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

  compare_bboxes(points)

