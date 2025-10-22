//-------------------------------------------------------------------------------------------
/*! \file    polygon_slice2.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
//-------------------------------------------------------------------------------------------
#include "polygon_slice2.h"
//-------------------------------------------------------------------------------------------
namespace trick
{

// Slice a polygon with vertical scanlines x=s along [xmin, xmax] with step.
// For each scanline, return only the outer (min_y, max_y) intersections.
// Notes:
//  - Works as silhouette for convex polygons.
//  - For concave polygons, min/max may bridge concavities.
template<int CN>
SliceResult SlicePolygon(
  const std::vector<cv::Vec<float,CN> > &contour_xy,
  float step,
  const std::pair<float,float> &x_range,
  float eps,
  float neighbor_eps)
{
  const int N = static_cast<int>(contour_xy.size());
  if (N < 3) {
    throw std::runtime_error("SlicePolygon: contour must have at least 3 points");
  }
  if (!(step > 0.0f)) {
    throw std::runtime_error("SlicePolygon: step must be positive");
  }

  // Determine scan range [xmin, xmax]
  float xmin = x_range.first;
  float xmax = x_range.second;
  if (std::isnan(xmin) || std::isnan(xmax)) {
    xmin = contour_xy[0][0];
    xmax = contour_xy[0][0];
    for (const auto &p : contour_xy) {
      xmin = std::min(xmin, p[0]);
      xmax = std::max(xmax, p[0]);
    }
  }

  // Number of scanlines (inclusive with tolerance)
  const int K = static_cast<int>(std::floor((xmax - xmin) / step + eps)) + 1;
  if (K <= 0) {
    return SliceResult{};
  }

  // Buckets of intersections per scanline: store (x, y)
  std::vector<std::vector<cv::Vec2f> > inters(K);

  // Iterate edges
  for (int i = 0; i < N; ++i) {
    const float x0 = contour_xy[i][0];
    const float y0 = contour_xy[i][1];
    const float x1 = contour_xy[(i + 1) % N][0];
    const float y1 = contour_xy[(i + 1) % N][1];

    const float dx = x1 - x0;
    const float dy = y1 - y0;

    // Skip edges parallel to scanline x=const (avoid overlap ambiguity)
    if (std::fabs(dx) < eps) continue;

    // Edge contributes to scanlines s with min(x0,x1) ≤ s < max(x0,x1)
    float ex_min = x0, ex_max = x1;
    if (ex_min > ex_max) std::swap(ex_min, ex_max);

    int k0 = static_cast<int>(std::ceil((ex_min - xmin) / step - eps));
    int k1 = static_cast<int>(std::floor((ex_max - xmin) / step - eps));  // upper end open
    if (k1 < 0 || k0 >= K) continue;
    k0 = std::max(k0, 0);
    k1 = std::min(k1, K - 1);

    for (int k = k0; k <= k1; ++k) {
      const float sx = xmin + k * step;     // scanline x
      const float t = (sx - x0) / dx;       // parameter on the edge
      if (t < -eps || t > 1.0f + eps) continue;
      const float y = y0 + t * dy;
      inters[k].emplace_back(sx, y);
    }
  }

  // For each scanline, keep only (min_y, max_y). Ignore too-close intersections.
  SliceResult out;
  out.pts1.reserve(K);
  out.pts2.reserve(K);

  for (int k = 0; k < K; ++k) {
    const auto &v = inters[k];
    if (v.size() < 2) continue;

    // Find min_y and max_y (no full sort needed)
    int i_min = 0, i_max = 0;
    float y_min = v[0][1], y_max = v[0][1];
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
      const float y = v[i][1];
      if (y < y_min) { y_min = y; i_min = i; }
      if (y > y_max) { y_max = y; i_max = i; }
    }
    if (std::fabs(y_max - y_min) < neighbor_eps) continue;

    out.pts1.push_back(v[i_min]);
    out.pts2.push_back(v[i_max]);
  }
  return out;
}
//-------------------------------------------------------------------------------------------

// Explicitly instantiate for CN=2 and CN=3
template SliceResult SlicePolygon<2>(
  const std::vector<cv::Vec<float, 2> >&,
  float, const std::pair<float,float> &, float, float);
template SliceResult SlicePolygon<3>(
  const std::vector<cv::Vec<float, 3> >&,
  float, const std::pair<float,float> &, float, float);
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------

