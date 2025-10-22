//-------------------------------------------------------------------------------------------
/*! \file    polygon_slice2_test.cpp
    \brief   Test code of polygon_slice2.cpp with the Python program.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025

#Compile:
g++ -std=c++17 -O3 -o polygon_slice2_test.out polygon_slice2_test.cpp polygon_slice2.cpp polygon_bb4.cpp `pkg-config --cflags --libs opencv4`
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "polygon_bb4.h"     // OBBResult, AngleMode, BoundingBoxWithEllipseAxis
#include "polygon_bb2.h"     // MinAreaRect (OBBResult)
#include "polygon_slice2.h"  // SlicePolygon (returns SliceResult)
//-------------------------------------------------------------------------------------------
namespace trick
{

// Rotation matrix for angle (radian): [[c, -s],[s, c]]
inline cv::Matx<float,2,2> RotXY(float angle_rad) {
  float c = std::cos(angle_rad);
  float s = std::sin(angle_rad);
  return cv::Matx<float,2,2>(c, -s,
                             s,  c);
}
//-------------------------------------------------------------------------------------------


struct SlicePipelineResult
{
  OBBResult obb;                 // center, size, angle
  std::string title;             // title string
  std::vector<cv::Vec2f> pts1;   // mapped back to original XY
  std::vector<cv::Vec2f> pts2;   // mapped back to original XY
};

// Rotate a list of points by R (2x2), using only XY
template<int CN>
inline std::vector<cv::Vec2f> RotatePoints(
  const std::vector<cv::Vec<float,CN>> &pts,
  const cv::Matx<float,2,2> &R)
{
  std::vector<cv::Vec2f> out;
  out.reserve(pts.size());
  for (const auto &p : pts) {
    float x = p[0], y = p[1];
    float xr = R(0,0)*x + R(0,1)*y;
    float yr = R(1,0)*x + R(1,1)*y;
    out.emplace_back(xr, yr);
  }
  return out;
}

// Map 2D points by R (2x2)
inline std::vector<cv::Vec2f> MapPoints(
  const std::vector<cv::Vec2f> &pts,
  const cv::Matx<float,2,2> &R)
{
  std::vector<cv::Vec2f> out;
  out.reserve(pts.size());
  for (const auto &p : pts) {
    float x = p[0], y = p[1];
    float xr = R(0,0)*x + R(0,1)*y;
    float yr = R(1,0)*x + R(1,1)*y;
    out.emplace_back(xr, yr);
  }
  return out;
}

// Convert Vec3f → Vec2f (use only XY)
inline std::vector<cv::Vec2f> ToVec2f(const std::vector<cv::Vec3f> &src) {
  std::vector<cv::Vec2f> dst;
  dst.reserve(src.size());
  for (const auto &p : src) {
    dst.emplace_back(p[0], p[1]);  // keep only XY
  }
  return dst;
}

// Main pipeline: choose OBB method, rotate, slice, map back
template<int CN>
inline SlicePipelineResult RunSlicePipeline(
  const std::vector<cv::Vec<float,CN>> &points,
  char choice_bb = '1',  // '1' -> MinAreaRect, otherwise BoundingBoxWithEllipseAxis
  AngleMode angle_mode = AngleMode::Symmetric,
  std::pair<float,float> s_range_p = {0.f, 1.f}) // percent range on the rotated x
{
  if (points.size() < 3) {
    throw std::runtime_error("RunSlicePipeline: need at least 3 points");
  }

  // Choose OBB
  OBBResult obb;
  std::string title;
  if (choice_bb == '1') {
    // MinAreaRect(points)
    if constexpr (CN == 2) obb = MinAreaRect(reinterpret_cast<const std::vector<cv::Vec2f>&>(points), angle_mode);
    else                   obb = MinAreaRect(ToVec2f(points), angle_mode);
    title = "SlicePolygon (MinAreaRect)";
  } else {
    // BoundingBoxWithEllipseAxis(points)
    if constexpr (CN == 2) obb = BoundingBoxWithEllipseAxis(reinterpret_cast<const std::vector<cv::Vec2f>&>(points), 0.0f, 1e-3, angle_mode);
    else                   obb = BoundingBoxWithEllipseAxis(reinterpret_cast<const std::vector<cv::Vec3f>&>(points), 0.0f, 1e-3, angle_mode);
    title = "SlicePolygon (BoundingBoxWithEllipseAxis)";
  }

  // Rotation matrices
  cv::Matx<float,2,2> Rneg = RotXY(-obb.angle);  // to rotate points to OBB frame
  cv::Matx<float,2,2> Rpos = RotXY( obb.angle);  // to map back (transpose of Rneg)

  // Rotate all points
  std::vector<cv::Vec2f> points_rot = RotatePoints(points, Rneg);

  // Determine scan range in rotated X
  float xmin = points_rot[0][0], xmax = points_rot[0][0];
  for (const auto &p : points_rot) {
    xmin = std::min(xmin, p[0]);
    xmax = std::max(xmax, p[0]);
  }
  float s_len = xmax - xmin;
  // Step: divide into ~50 bands; guard for zero length
  float step = (s_len > 0.0f) ? (s_len / 50.0f) : 1.0f;

  // Apply percent range
  float s0 = xmin + s_len * s_range_p.first;
  float s1 = xmin + s_len * s_range_p.second;

  // Slice on rotated points
  SliceResult sr = SlicePolygon(points_rot, step, {s0, s1});

  // Map back to original XY
  SlicePipelineResult res;
  res.obb = obb;
  res.title = title;
  res.pts1 = MapPoints(sr.pts1, Rpos);
  res.pts2 = MapPoints(sr.pts2, Rpos);
  return res;
}
//-------------------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void PrintResultAsJSON(const SlicePipelineResult &r)
{
  const auto &obb = r.obb;

  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(9);

  std::cout << "{";
  std::cout << "\"center\":[" << obb.center.x << "," << obb.center.y << "],";
  std::cout << "\"size\":[" << obb.size.width << "," << obb.size.height << "],";
  std::cout << "\"angle\":" << obb.angle << ",";
  std::cout << "\"title\":\"" << r.title << "\",";

  // pts1
  std::cout << "\"pts1\":[";
  for (size_t i = 0; i < r.pts1.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << "[" << r.pts1[i][0] << "," << r.pts1[i][1] << "]";
  }
  std::cout << "],";

  // pts2
  std::cout << "\"pts2\":[";
  for (size_t i = 0; i < r.pts2.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << "[" << r.pts2[i][0] << "," << r.pts2[i][1] << "]";
  }
  std::cout << "]";

  std::cout << "}\n";
}

int main(int argc, char**argv)
{
  char choice_bb = (argc > 1) ? argv[1][0] : '1';  // '1' -> MinAreaRect, otherwise BoundingBoxWithEllipseAxis
  std::pair<float,float> s_range_p = {0.f, 1.f};  // percent range on the rotated x
  s_range_p.first = (argc > 2) ? std::stof(argv[2]) : 0.f;
  s_range_p.second = (argc > 3) ? std::stof(argv[3]) : 1.f;

  std::vector<cv::Vec3f> pts;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    float x, y, z = 0.0f;
    if (!(ss >> x >> y)) continue;
    ss >> z;  // optional
    pts.emplace_back(x, y, z);
  }

  if (pts.empty()) {
    std::cerr << "No valid points received.\n";
    return 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  // '1' -> MinAreaRect, otherwise BoundingBoxWithEllipseAxis
  auto r = RunSlicePipeline(pts, choice_bb, AngleMode::Symmetric, s_range_p);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  PrintResultAsJSON(r);

  std::cerr << "RunSlicePipeline(C++, "<<r.title<<") computed with "<<ms<<" ms"<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
