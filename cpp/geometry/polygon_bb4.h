//-------------------------------------------------------------------------------------------
/*! \file    polygon_bb4.h
    \brief   Port of polygon_bb4.py: Polygon bounding box detection with ellipse axis estimation.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025
*/
//-------------------------------------------------------------------------------------------
#ifndef polygon_bb4_h
#define polygon_bb4_h
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <vector>
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

/*
Angle constraint mode:
  amPositive: [0,pi]
  amSymmetric: [-pi/2,pi/2]
  amNone: No constraint.
*/
enum class TAngleMode { amNone=0, amSymmetric, amPositive };

// Oriented Bounding Box.
// Similar to cv::RotatedRect, but the angle is in radian.
struct TOrientedBB
{
  cv::Point2f center {0.f, 0.f};
  cv::Size2f size {0.f, 0.f};    // size.width is always greater than (or equal to) size.height.
  float angle  {0.f};            // angle is in radian.  angle direction is always the longer side direction (undefined for a square case).
  TOrientedBB() = default;
  TOrientedBB(const cv::Point2f &c, const cv::Size2f &s, const float &a)
    : center(c), size(s), angle(a)  {}
};

//-------------------------------------------------------------------------------------------

/*
Convert angle (radian) to the range according to mode:
  mode=amPositive: [0,pi]
  mode=amSymmetric: [-pi/2,pi/2]
  mode=amNone: No modification.
*/
inline float AngleModHalf(float q, TAngleMode mode)
{
  const float PI = static_cast<float>(CV_PI);
  if (mode == TAngleMode::amSymmetric)
  {
    // Normalize to [-pi/2, pi/2)
    float r = std::fmod(q + 0.5f*PI, PI);
    if (r < 0.0f) r += PI;
    r -= 0.5f*PI;
    return r;
  }
  else if (mode == TAngleMode::amPositive)
  {
    // Normalize to [0, pi)
    float r = std::fmod(q, PI);
    if (r < 0.0f) r += PI;
    return r;
  }
  return q;  // TAngleMode::amNone
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
  NOTE: This function is instantiated only for CN = 2, 3.
*/
template<int CN>
TOrientedBB BoundingBoxWithEllipseAxis(
  const std::vector<cv::Vec<float, CN> > &pts,
  float trim_p = 0.0f,
  float ridge = 1e-3f,
  TAngleMode angle_mode = TAngleMode::amSymmetric);

//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // polygon_bb4_h
//-------------------------------------------------------------------------------------------
