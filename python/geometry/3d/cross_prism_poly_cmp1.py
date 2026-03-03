#!/usr/bin/python3
#\file    cross_prism_poly_cmp1.py
#\brief   Test code of detecting intersection between cross-prism 3d (convex) and polygon.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.19, 2026
import numpy as np
import time
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- 1. Geometry Generation ---

def generate_random_scene():
  """
  Generates a random 3D scene consisting of a convex polyhedron (representing the robot finger)
  and a 3D rectangular polygon (representing the container wall).
  """
  # Generate a random approximated Convex CrossPrism (12-15 vertices)
  # The points are scaled to represent a realistic finger size (~20cm).
  prism_pts = np.random.rand(12, 3) * 0.2
  hull = ConvexHull(prism_pts)
  prism_hull_pts = prism_pts[hull.vertices]

  # Generate a random 3D Rectangle (Polygon) representing a container wall
  center = np.random.rand(3) * 0.3
  u = np.random.randn(3)
  v = np.random.randn(3)
  # Orthogonalize and normalize the basis vectors to form a perfect rectangle
  u /= np.linalg.norm(u)
  v -= np.dot(v, u) * u
  v /= np.linalg.norm(v)

  size_u, size_v = 0.15, 0.15
  poly_pts = np.array([
    center + size_u * u + size_v * v,
    center - size_u * u + size_v * v,
    center - size_u * u - size_v * v,
    center + size_u * u - size_v * v
  ])

  # Sometimes force the shapes closer to each other to increase the probability of collision during tests.
  if np.random.rand() > 0.5:
    prism_hull_pts += (np.mean(poly_pts, axis=0) - np.mean(prism_hull_pts, axis=0)) * np.random.rand()

  return prism_hull_pts, poly_pts

# --- 2. Algorithms ---

def check_sat(prism_pts, poly_pts):
  """
  Separating Axis Theorem (SAT) implementation.
  Checks if there is any axis along which the projections of the two convex shapes do not overlap.
  """
  start_time = time.perf_counter()
  eps = 1e-6 # Epsilon for floating-point margin to prevent false positives

  hull = ConvexHull(prism_pts)
  prism_normals = hull.equations[:, :3]

  # Calculate the normal vector of the container wall (polygon)
  poly_v1 = poly_pts[1] - poly_pts[0]
  poly_v2 = poly_pts[2] - poly_pts[1]
  poly_normal = np.cross(poly_v1, poly_v2)
  poly_normal_norm = np.linalg.norm(poly_normal)
  if poly_normal_norm < eps:
    return False, time.perf_counter() - start_time
  poly_normal /= poly_normal_norm

  def get_unique_edges(pts, simplices=None):
    """
    Helper function to extract unique edges from a set of points or a mesh.
    This drastically reduces the number of cross-product calculations by removing duplicate or parallel edges.
    """
    edges = []
    if simplices is not None:
      for simplex in simplices:
        for i in range(3):
          edge = pts[simplex[(i+1)%3]] - pts[simplex[i]]
          norm = np.linalg.norm(edge)
          if norm > eps:
            edge = edge / norm
            # Check if an edge with the same (or exactly opposite) direction already exists
            if not any(np.abs(np.dot(e, edge)) > 1.0 - eps for e in edges):
              edges.append(edge)
    else:
      num_pts = len(pts)
      for i in range(num_pts):
        edge = pts[(i+1)%num_pts] - pts[i]
        norm = np.linalg.norm(edge)
        if norm > eps:
          edge = edge / norm
          if not any(np.abs(np.dot(e, edge)) > 1.0 - eps for e in edges):
            edges.append(edge)
    return edges

  prism_edges = get_unique_edges(prism_pts, hull.simplices)
  poly_edges = get_unique_edges(poly_pts)

  # The candidate separating axes are:
  # 1. Face normals of shape A
  # 2. Face normals of shape B
  # 3. Cross products of all unique edge combinations between shape A and shape B
  axes = list(prism_normals) + [poly_normal]
  for pe in poly_edges:
    for pre in prism_edges:
      cross_prod = np.cross(pe, pre)
      norm = np.linalg.norm(cross_prod)
      if norm > eps:
        axes.append(cross_prod / norm)

  # Test overlaps on all candidate axes
  collision = True
  for axis in axes:
    proj_prism = np.dot(prism_pts, axis)
    proj_poly = np.dot(poly_pts, axis)
    # If there is a gap between the min/max projections, a separating axis is found.
    if np.max(proj_prism) < np.min(proj_poly) - eps or np.max(proj_poly) < np.min(proj_prism) - eps:
      collision = False
      break

  return collision, time.perf_counter() - start_time

def check_gjk(prism_pts, poly_pts):
  """
  Gilbert-Johnson-Keerthi (GJK) collision detection algorithm.
  Determines whether the Minkowski difference of two convex shapes contains the origin.
  """
  start_time = time.perf_counter()

  # Calculate the normal of the 2D polygon
  poly_v1 = poly_pts[1] - poly_pts[0]
  poly_v2 = poly_pts[2] - poly_pts[1]
  poly_normal = np.cross(poly_v1, poly_v2)
  poly_normal_norm = np.linalg.norm(poly_normal)
  if poly_normal_norm < 1e-6:
    return False, time.perf_counter() - start_time
  poly_normal /= poly_normal_norm

  # CRITICAL FIX: Add virtual thickness to the 2D polygon.
  # GJK requires volume. A perfectly flat 2D polygon causes numerical degeneracy
  # when calculating the Minkowski difference in 3D space.
  thickness = 1e-4
  thick_poly_pts = np.vstack([poly_pts + poly_normal * thickness, poly_pts - poly_normal * thickness])

  def support(pts, d):
    # Returns the furthest point in the given direction 'd'
    return pts[np.argmax(np.dot(pts, d))]

  def get_support(d):
    # Support function for the Minkowski difference: Supp(A) - Supp(B, -d)
    return support(prism_pts, d) - support(thick_poly_pts, -d)

  # Initialize the simplex with a random search direction
  d = np.array([1.0, 0.0, 0.0])
  simplex = [get_support(d)]
  d = -simplex[0] # Next search direction is towards the origin

  collision = False
  for _ in range(64): # Cap iterations to prevent infinite loops in degenerate cases
    if np.linalg.norm(d) < 1e-6:
      collision = True
      break

    a = get_support(d)
    # If the newly added point hasn't passed the origin, the origin cannot be enclosed.
    if np.dot(a, d) < 0:
      collision = False
      break

    simplex.append(a)

    # Simplex resolution logic: Update the simplex and determine the next search direction
    if len(simplex) == 2:
      b, a = simplex
      ab = b - a
      ao = -a
      if np.dot(ab, ao) > 0:
        d = np.cross(np.cross(ab, ao), ab)
      else:
        simplex = [a]
        d = ao
    elif len(simplex) == 3:
      c, b, a = simplex
      ab, ac, ao = b - a, c - a, -a
      abc = np.cross(ab, ac)
      if np.dot(np.cross(abc, ac), ao) > 0:
        if np.dot(ac, ao) > 0:
          simplex = [c, a]
          d = np.cross(np.cross(ac, ao), ac)
        else:
          simplex = [b, a] if np.dot(ab, ao) > 0 else [a]
          d = np.cross(np.cross(ab, ao), ab) if np.dot(ab, ao) > 0 else ao
      else:
        if np.dot(np.cross(ab, abc), ao) > 0:
          simplex = [b, a] if np.dot(ab, ao) > 0 else [a]
          d = np.cross(np.cross(ab, ao), ab) if np.dot(ab, ao) > 0 else ao
        else:
          d = abc if np.dot(abc, ao) > 0 else -abc
          if np.dot(abc, ao) <= 0: # Origin is enclosed by the triangle (rare but possible)
            simplex = [c, b, a]
    elif len(simplex) == 4:
      d_pt, c, b, a = simplex
      ab, ac, ad, ao = b - a, c - a, d_pt - a, -a
      abc, acd, adb = np.cross(ab, ac), np.cross(ac, ad), np.cross(ad, ab)

      # Check which face of the tetrahedron the origin is facing
      if np.dot(abc, ao) > 0:
        simplex, d = [c, b, a], abc
      elif np.dot(acd, ao) > 0:
        simplex, d = [d_pt, c, a], acd
      elif np.dot(adb, ao) > 0:
        simplex, d = [b, d_pt, a], adb
      else:
        # If the origin is not outside any of the 3 faces, it must be inside the tetrahedron!
        collision = True
        break

  return collision, time.perf_counter() - start_time

def check_approximation(prism_pts, poly_pts):
  """
  A fast, approximate collision detection algorithm between a 3D Convex Polyhedron (Finger)
  and a Planar Convex Polygon (Container Wall).

  It computes the 2D cross-section of the polyhedron on the polygon's plane, and checks
  if any vertex of this cross-section lies inside the target polygon.

  [WARNING] Known Failure Cases (False Negatives):
  This algorithm will fail to detect a collision in the following topological conditions:
    1. Complete Enclosure: The polyhedron's cross-section completely encloses the target
       polygon (no cross-section vertices are inside the target polygon's boundary).
    2. Edge-only Intersection: The cross-section and the polygon intersect only at their
       edges (e.g., overlapping like a star shape) without any vertex falling inside
       the other's area.
  """
  start_time = time.perf_counter()
  eps = 1e-6

  # Step 1: Extract the infinite plane equation (Normal and Distance) from the Polygon
  v1 = poly_pts[1] - poly_pts[0]
  v2 = poly_pts[2] - poly_pts[1]
  normal = np.cross(v1, v2)
  normal_norm = np.linalg.norm(normal)
  if normal_norm < eps:
    return False, time.perf_counter() - start_time, []
  normal /= normal_norm
  d_plane = -np.dot(normal, poly_pts[0]) # Plane equation: N dot P + d = 0

  # Step 2: Broad-phase check.
  # Check the signed distance of all prism vertices to the plane.
  distances = np.dot(prism_pts, normal) + d_plane
  # If ALL vertices are strictly on the positive side, or strictly on the negative side,
  # the prism does not intersect the infinite plane at all. (Fast exit)
  if np.all(distances > eps) or np.all(distances < -eps):
    return False, time.perf_counter() - start_time, []

  # Step 3: Edge-Plane intersection calculation.
  hull = ConvexHull(prism_pts)
  intersect_pts = []
  unique_edges = set()

  # Store edge vertex indices in a Set to prevent evaluating shared edges multiple times
  for simplex in hull.simplices:
    for i in range(3):
      idx1 = simplex[i]
      idx2 = simplex[(i+1)%3]
      unique_edges.add((min(idx1, idx2), max(idx1, idx2)))

  for idx1, idx2 in unique_edges:
    d1 = distances[idx1]
    d2 = distances[idx2]

    # If the signed distances have opposite signs, the edge crosses the plane.
    # We include <= eps to catch edges that are exactly touching the plane.
    if d1 * d2 <= eps and abs(d2 - d1) > eps:
      p1 = prism_pts[idx1]
      p2 = prism_pts[idx2]
      # Linear interpolation to find the exact 3D intersection point
      t = -d1 / (d2 - d1)
      if 0.0 <= t <= 1.0:
        intersect_pts.append(p1 + t * (p2 - p1))

  if not intersect_pts:
    return False, time.perf_counter() - start_time, []

  # Reverse Check
  equations = hull.equations
  for poly_pt in poly_pts:
    pt_homo = np.append(poly_pt, 1.0)
    if np.all(np.dot(equations, pt_homo) <= eps):
      return True, time.perf_counter() - start_time, intersect_pts

  # Step 4: Robust 3D Point-in-Convex-Polygon Check.
  # For each intersection point on the plane, check if it falls inside the boundaries of the container wall.
  num_poly_pts = len(poly_pts)
  for pt in intersect_pts:
    is_inside = True
    for i in range(num_poly_pts):
      edge_vec = poly_pts[(i+1)%num_poly_pts] - poly_pts[i]
      pt_vec = pt - poly_pts[i]

      # The cross product of the polygon edge and the vector to the point
      # should point in the same direction as the polygon's normal if the point is inside.
      cross_p = np.cross(edge_vec, pt_vec)
      if np.dot(cross_p, normal) < -eps:
        is_inside = False
        break

    if is_inside:
      return True, time.perf_counter() - start_time, intersect_pts

  return False, time.perf_counter() - start_time, []

# --- 3. Visualization ---

def plot_scene(prism_pts, poly_pts, is_collision, intersect_pts=None):
  """
  Renders the 3D scene using Matplotlib.
  Highlights the container wall in Red if a collision is detected.
  Also plots the exact intersection points on the wall.
  """
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')

  # Render the Robot Finger (CrossPrism) as a translucent blue mesh
  hull = ConvexHull(prism_pts)
  for s in hull.simplices:
    tri = Poly3DCollection([prism_pts[s]])
    tri.set_color('cyan')
    tri.set_alpha(0.3)
    tri.set_edgecolor('blue')
    ax.add_collection3d(tri)

  # Render the Container Wall (Polygon). Red = Collision, Green = Safe
  poly_color = 'red' if is_collision else 'green'
  poly = Poly3DCollection([poly_pts])
  poly.set_color(poly_color)
  poly.set_alpha(0.6)
  poly.set_edgecolor('dark' + poly_color)
  ax.add_collection3d(poly)

  # If a collision is detected, plot the intersection points.
  if is_collision and intersect_pts:
    pts_arr = np.array(intersect_pts)
    ax.scatter(pts_arr[:,0], pts_arr[:,1], pts_arr[:,2],
               color='green', s=100, marker='X', zorder=5, label='Intersection Points')
    ax.legend()

  # Auto-scale the axes to fit all objects perfectly
  all_pts = np.vstack((prism_pts, poly_pts))
  ax.set_xlim([np.min(all_pts[:,0]), np.max(all_pts[:,0])])
  ax.set_ylim([np.min(all_pts[:,1]), np.max(all_pts[:,1])])
  ax.set_zlim([np.min(all_pts[:,2]), np.max(all_pts[:,2])])

  status_text = "COLLISION DETECTED" if is_collision else "SAFE (NO COLLISION)"
  ax.set_title(f"Interference Check: {status_text}", fontsize=14, color=poly_color)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()

# --- 4. Main Execution ---

def run_tests():
  """
  Main function to generate a random scene, evaluate it using all three algorithms,
  compare execution times, and render the result.
  """
  seed = int(time.time())
  np.random.seed(seed)
  print(f"\n=== Random Seed: {seed} ===")

  prism_pts, poly_pts = generate_random_scene()

  res_sat, t_sat = check_sat(prism_pts, poly_pts)
  res_gjk, t_gjk = check_gjk(prism_pts, poly_pts)
  res_opt, t_opt, intersect_pts_opt = check_approximation(prism_pts, poly_pts)

  print(f"{'Algorithm':<15} | {'Result':<10} | {'Time (ms)':<15}")
  print("-" * 45)
  print(f"{'SAT':<15} | {str(res_sat):<10} | {t_sat*1000:.4f}")
  print(f"{'GJK':<15} | {str(res_gjk):<10} | {t_gjk*1000:.4f}")
  print(f"{'Optimized':<15} | {str(res_opt):<10} | {t_opt*1000:.4f}")

  # Ensure all algorithms agree on the collision result
  if not (res_sat == res_gjk == res_opt):
    print("\nWARNING: Algorithms returned inconsistent results!")

  if res_opt and intersect_pts_opt:
    print("\n--- Intersection Points (from Optimized Algorithm) ---")
    for i, pt in enumerate(intersect_pts_opt):
      print(f"Point {i+1}: [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]")

  # Render the scene using Matplotlib
  plot_scene(prism_pts, poly_pts, res_opt, intersect_pts_opt)

if __name__ == "__main__":
  run_tests()

