//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat7.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2016

g++ -g -Wall -O2 -o cv2-mat7.out cv2-mat7.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  float array[]= {1.0,2.0,3.0};
  cv::Mat m1(1,3,CV_32F, array);
  print(m1);

  array[1]*= 100.0;
  print(m1);

  return 0;
}
//-------------------------------------------------------------------------------------------
