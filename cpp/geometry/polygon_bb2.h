//-------------------------------------------------------------------------------------------
/*! \file    polygon_bb2.h
    \brief   Port of polygon_bb2.py: Polygon bounding box detection with Minimum Area Rect fitting.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025
*/
//-------------------------------------------------------------------------------------------
#ifndef polygon_bb2_h
#define polygon_bb2_h
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "polygon_bb4.h"
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

/*
Get a rectangle of minimum area.
  angle_mode: Angle normalization mode ('positive': [0,pi], 'symmetric': [-pi/2,pi/2], None).
Return: center,size,angle.
  angle is in radian.  angle direction is always the longer side direction.
  size[0] is always greater than size[1].
*/
inline OBBResult MinAreaRect(
  const std::vector<cv::Vec2f> &pts,
  AngleMode angle_mode = AngleMode::Symmetric)
{
  if (pts.empty()) {
    throw std::runtime_error("MinAreaRect: Empty input");
  }
  if (pts.size() == 1) {
    return OBBResult{cv::Point2f(pts[0][0], pts[0][1]), cv::Size2f(0.f, 0.f), 0.f};
  }

  // Wrap as Mat without copy (N x 1, CV_32FC2). const_cast is safe here: minAreaRect won't modify data.
  cv::Mat pts_view(static_cast<int>(pts.size()), 1, CV_32FC2,
                   const_cast<cv::Vec2f*>(pts.data()));

  // Compute minimum-area rectangle
  cv::RotatedRect rr = cv::minAreaRect(pts_view);

  // OpenCV angle is in degrees, range [-90, 0)
  float angle = rr.angle * static_cast<float>(CV_PI / 180.0f);

  // Ensure size.width >= size.height and angle follows the long side
  const bool flip = (rr.size.width < rr.size.height);
  const float w = flip ? rr.size.height : rr.size.width;
  const float h = flip ? rr.size.width  : rr.size.height;
  if (flip) angle += static_cast<float>(CV_PI * 0.5f);

  // Normalize angle
  angle = AngleModHalf(angle, angle_mode);

  OBBResult out;
  out.center = rr.center;
  out.size   = cv::Size2f(w, h);
  out.angle  = angle;
  return out;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // polygon_bb2_h
//-------------------------------------------------------------------------------------------
