//-------------------------------------------------------------------------------------------
/*! \file    polygon_bb4.cpp
    \brief   Port of polygon_bb4.py: Polygon bounding box detection with ellipse axis estimation.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
//-------------------------------------------------------------------------------------------
//#define USE_EIGEN_SOLVER
#ifdef USE_EIGEN_SOLVER
  #include <Eigen/Dense>
#endif
//-------------------------------------------------------------------------------------------
#include "polygon_bb4.h"
//-------------------------------------------------------------------------------------------
namespace trick
{

// Fast percentile using order statistic index (no interpolation)
// p in [0,100], returns element at round(p/100*(N-1))
template<typename T>
inline T PercentileByIndex(std::vector<T> v, float p) {
  const int n = static_cast<int>(v.size());
  if (n == 0) return T(0);
  if (p <= 0.0f) {
    return *std::min_element(v.begin(), v.end());
  } else if (p >= 100.0f) {
    return *std::max_element(v.begin(), v.end());
  }
  const float pos = p * 0.01f * static_cast<float>(n - 1);
  int k = static_cast<int>(std::round(pos));
  k = std::max(0, std::min(n - 1, k));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}
//-------------------------------------------------------------------------------------------

/*
Oriented bounding box whose long edge is parallel to the ellipse major axis.
  XY: (N,2) float-like.
  trim_p: Percentile trimming to remove the outliers (should be in [0.0, 100.0]).
  ridge: Ridge parameter for numerical stability.
  angle_mode: Angle normalization mode ('positive': [0,pi], 'symmetric': [-pi/2,pi/2], None).
  Return: center, size, angle
    center = (cx_w, cy_w)
    size = (w, h): w >= h
    angle : Angle direction is always the major (longer) axis direction. Normalized with angle_mode.
*/
template<int CN>
inline OBBResult BoundingBoxWithEllipseAxis(
    const std::vector<cv::Vec<float, CN> > &pts,
    float trim_p,
    double ridge,
    AngleMode angle_mode)
{
  const int n = static_cast<int>(pts.size());
  if (n == 0) {
    throw std::runtime_error("BoundingBoxWithEllipseAxis: Empty input");
  }
  if (n == 1) {
    return OBBResult{cv::Point2f(pts[0][0], pts[0][1]), cv::Size2f(0.f, 0.f), 0.f};
  }

  // -- 1) Centroid (simple mean for speed) --
  double sumx = 0.0, sumy = 0.0;
  for (const auto &p : pts) {
    sumx += static_cast<double>(p[0]);
    sumy += static_cast<double>(p[1]);
  }
  const double cx = sumx / n;
  const double cy = sumy / n;

  // -- 2) Accumulate normal equations for linear system (5 unknowns) --
  // We fit: [xx, xy, yy, x, y] * param ≈ 1 (F = -1). params = [A,B,C,D,E]^T
  // AtA is 5x5, Atb is 5x1
#ifdef USE_EIGEN_SOLVER
  Eigen::Matrix<double,5,5> AtA = Eigen::Matrix<double,5,5>::Zero();
  Eigen::Matrix<double,5,1> Atb = Eigen::Matrix<double,5,1>::Zero();
  for (const auto &p : pts) {
    double dx = double(p[0]) - cx, dy = double(p[1]) - cy;
    double a0 = dx*dx, a1 = dx*dy, a2 = dy*dy, a3 = dx, a4 = dy;
    Eigen::Matrix<double,5,1> a; a << a0,a1,a2,a3,a4;
    AtA.noalias() += a * a.transpose();
    Atb.noalias() += a;
  }
  AtA.diagonal().array() += ridge;
  Eigen::LLT<Eigen::Matrix<double,5,5> > llt(AtA);
  Eigen::Matrix<double,5,1> x = llt.solve(Atb);
  cv::Vec<double,5> param; for (int i=0;i<5;++i) param[i]= x(i);

#else
  cv::Matx<double,5,5> AtA = cv::Matx<double,5,5>::zeros();
  cv::Vec<double,5>    Atb = cv::Vec<double,5>::all(0);

  for (const auto &p : pts) {
    const double dx = static_cast<double>(p[0]) - cx;
    const double dy = static_cast<double>(p[1]) - cy;
    const double xx = dx*dx;
    const double xy = dx*dy;
    const double yy = dy*dy;

    double a[5] = {xx, xy, yy, dx, dy};

    // AtA += a^T a
    // (fill only upper triangle then mirror for speed/readability)
    for (int r = 0; r < 5; ++r) {
      for (int c = r; c < 5; ++c) {
        AtA(r,c) += a[r] * a[c];
      }
    }
    // Atb += a * 1.0
    for (int r = 0; r < 5; ++r) {
      Atb[r] += a[r];
    }
  }
  // Symmetrize AtA and add ridge
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < r; ++c) {
      AtA(r,c) = AtA(c,r);
    }
    AtA(r,r) += ridge;
  }

  // -- Solve (AtA) * p = Atb --
  cv::Vec<double,5> param;
  cv::Matx<double,5,1> x;
  bool ok = cv::solve(AtA, cv::Matx<double,5,1>(Atb), x, cv::DECOMP_CHOLESKY);
  if (!ok) {
    // Fallback to LDLT if needed
    ok = cv::solve(AtA, cv::Matx<double,5,1>(Atb), x, cv::DECOMP_LU);
    if (!ok) throw std::runtime_error("Linear solve failed");
  }
  for (int i = 0; i < 5; ++i) param[i] = x(i,0);
#endif

  const double Acoef = param[0];
  const double Bcoef = param[1];
  const double Ccoef = param[2];

  // -- 3) Major-axis angle from conic (ef6's ellipse_angle_of_rotation) --
  const double b_half = 0.5 * Bcoef;
  double angle = 0.0;
  if (std::abs(b_half) < 1e-15) {
    angle = (Acoef > Ccoef) ? 0.0 : (0.5 * CV_PI);
  } else {
    const double denom = (Acoef - Ccoef);
    if (Acoef > Ccoef) {
      angle = 0.5 * std::atan2(2.0 * b_half, denom);
    } else {
      angle = 0.5 * CV_PI + 0.5 * std::atan2(2.0 * b_half, denom);
    }
  }

  // -- 5) Project to candidate major/minor axes (no explicit rotation matrix) --
  const double ca = std::cos(angle);
  const double sa = std::sin(angle);

  std::vector<float> proj_x; proj_x.reserve(n);
  std::vector<float> proj_y; proj_y.reserve(n);

  for (const auto &p : pts) {
    const double pdx = static_cast<double>(p[0]) - cx;
    const double pdy = static_cast<double>(p[1]) - cy;
    const double px = pdx * ca + pdy * sa;   // along major candidate
    const double py = -pdx * sa + pdy * ca;  // along minor
    proj_x.push_back(static_cast<float>(px));
    proj_y.push_back(static_cast<float>(py));
  }

  // -- 6) Percentile trimming (optional) --
  float x_lo, x_hi, y_lo, y_hi;
  if (trim_p > 0.0f && trim_p < 100.0f) {
    x_lo = PercentileByIndex(proj_x, trim_p);
    x_hi = PercentileByIndex(proj_x, 100.0f - trim_p);
    y_lo = PercentileByIndex(proj_y, trim_p);
    y_hi = PercentileByIndex(proj_y, 100.0f - trim_p);
  } else {
    auto [minx_it, maxx_it] = std::minmax_element(proj_x.begin(), proj_x.end());
    auto [miny_it, maxy_it] = std::minmax_element(proj_y.begin(), proj_y.end());
    x_lo = *minx_it; x_hi = *maxx_it;
    y_lo = *miny_it; y_hi = *maxy_it;
  }

  float w = x_hi - x_lo;
  float h = y_hi - y_lo;

  // -- 7) Center in world coords: midpoint in rotated frame -> unrotate + shift --
  const float mid_x = 0.5f * (x_lo + x_hi);
  const float mid_y = 0.5f * (y_lo + y_hi);
  const float cx_w = static_cast<float>(cx + mid_x * ca - mid_y * sa);
  const float cy_w = static_cast<float>(cy + mid_x * sa + mid_y * ca);

  // -- 8) Enforce "angle is the long-edge direction" --
  if (h > w) {
    angle += 0.5 * CV_PI;
    std::swap(w, h);
  }

  float angle_f = AngleModHalf(static_cast<float>(angle), angle_mode);

  OBBResult out;
  out.center = cv::Point2f(cx_w, cy_w);
  out.size   = cv::Size2f(w, h);
  out.angle  = angle_f;
  return out;
}
//-------------------------------------------------------------------------------------------

// Explicitly instantiate for CN=2 and CN=3
template OBBResult BoundingBoxWithEllipseAxis<2>(
  const std::vector<cv::Vec<float, 2> >&, float, double, AngleMode);
template OBBResult BoundingBoxWithEllipseAxis<3>(
  const std::vector<cv::Vec<float, 3> >&, float, double, AngleMode);
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------






