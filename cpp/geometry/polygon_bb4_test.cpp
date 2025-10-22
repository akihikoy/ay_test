//-------------------------------------------------------------------------------------------
/*! \file    polygon_bb4_test.cpp
    \brief   Test code of polygon_bb4.cpp with the Python program.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.22, 2025

#Compile with Eigen solver and OpenCV:
g++ -std=c++17 -O3 -o polygon_bb4_test.out polygon_bb4_test.cpp polygon_bb4.cpp -I/usr/include/eigen3 `pkg-config --cflags --libs opencv4`

#Test:
./polygon_bb4_test.py
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/core.hpp>
#include <chrono>
#include "polygon_bb4.h"
//-------------------------------------------------------------------------------------------
namespace trick
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::vector<cv::Vec3f> pts;
  std::string line;
  while (std::getline(std::cin, line))
  {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    float x, y, z = 0.0f;
    if (!(ss >> x >> y)) continue;
    ss >> z;  // optional
    pts.emplace_back(x, y, z);
  }

  if (pts.empty())
  {
    std::cerr << "No valid points received.\n";
    return 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto r = BoundingBoxWithEllipseAxis(pts, 0.0f, 1e-3f, TAngleMode::amSymmetric);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout.setf(std::ios::fixed);
  std::cout.precision(9);
  std::cout << r.center.x << " " << r.center.y << " "
            << r.size.width << " " << r.size.height << " "
            << r.angle << std::endl;

  std::cerr << "BoundingBoxWithEllipseAxis(C++) computed with "<<ms<<" ms"<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
