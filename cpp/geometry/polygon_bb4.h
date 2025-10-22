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

enum class AngleMode { Symmetric, Positive, None };

struct OBBResult {
  cv::Point2f center;
  cv::Size2f size;
  float angle;
};

template<int CN>
OBBResult BoundingBoxWithEllipseAxis(
  const std::vector<cv::Vec<float, CN> > &pts,
  float trim_p = 0.0f,
  double ridge = 1e-3,
  AngleMode angle_mode = AngleMode::Symmetric);

//-------------------------------------------------------------------------------------------

// Convert angle (rad) according to AngleMode half-period normalization
inline float AngleModHalf(float q, AngleMode mode)
{
  const float PI = static_cast<float>(CV_PI);
  if (mode == AngleMode::Symmetric)
  {
    // Normalize to [-pi/2, pi/2)
    float r = std::fmod(q + 0.5f*PI, PI);
    if (r < 0.0f) r += PI;
    r -= 0.5f*PI;
    return r;
  }
  else if (mode == AngleMode::Positive)
  {
    // Normalize to [0, pi)
    float r = std::fmod(q, PI);
    if (r < 0.0f) r += PI;
    return r;
  }
  return q;  // AngleMode::None
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // polygon_bb4_h
//-------------------------------------------------------------------------------------------
