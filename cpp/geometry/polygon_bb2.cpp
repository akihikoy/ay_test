//-------------------------------------------------------------------------------------------
/*! \file    polygon_bb2.cpp
    \brief   Port of polygon_bb2.py: Polygon bounding box detection with Minimum Area Rect fitting.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025

#Compile with cv::solve:
g++ -std=c++17 -O3 -o polygon_bb2.out polygon_bb2.cpp `pkg-config --cflags --libs opencv4`

#Test (same as polygon_bb4):
./polygon_bb4_test.py
*/
//-------------------------------------------------------------------------------------------
// #include <algorithm>
// #include <vector>
// #include <cmath>
#include <iostream>
#include <sstream>
//-------------------------------------------------------------------------------------------
#include "polygon_bb2.h"
//-------------------------------------------------------------------------------------------
namespace trick
{

//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------

using namespace trick;

int main(int argc, char**argv)
{
  std::vector<cv::Vec2f> pts;
  std::string line;
  while (std::getline(std::cin, line))
  {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    float x, y, z = 0.0f;
    if (!(ss >> x >> y)) continue;
    ss >> z;  // optional
    pts.emplace_back(x, y);
  }

  if (pts.empty())
  {
    std::cerr << "No valid points received.\n";
    return 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto r = MinAreaRect(pts, TAngleMode::amSymmetric);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout.setf(std::ios::fixed);
  std::cout.precision(9);
  std::cout << r.center.x << " " << r.center.y << " "
            << r.size.width << " " << r.size.height << " "
            << r.angle << std::endl;

  std::cerr << "MinAreaRect(C++) computed with "<<ms<<" ms"<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
