//-------------------------------------------------------------------------------------------
/*! \file    polygon_slice2.h
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025
*/
//-------------------------------------------------------------------------------------------
#ifndef polygon_slice2_h
#define polygon_slice2_h
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <vector>
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

// Return type of SlicePolygon.
struct SliceResult
{
  std::vector<cv::Vec2f> pts1;  // lower y (min)
  std::vector<cv::Vec2f> pts2;  // upper y (max)
};

/*
Slice a polygon for scanlines (x_range, step) along the x-axis.
  - For each scanline, only the outer two intersections are returned.
    - Works as silhouette for convex polygons.
    - For concave polygons, min/max may bridge concavities.
  NOTE: This function is instantiated only for CN = 2, 3.
*/
template<int CN>
SliceResult SlicePolygon(
  const std::vector<cv::Vec<float,CN> > &contour_xy,
  float step,
  const std::pair<float,float> &x_range = {NAN, NAN},
  float eps = 1e-12f,
  float neighbor_eps = 1e-9f);
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // polygon_slice2_h
//-------------------------------------------------------------------------------------------
