//-------------------------------------------------------------------------------------------
/*! \file    cv2-mat_elem.cpp
    \brief   Test element access of cv::Mat.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.16, 2021

g++ -g -Wall -O2 -o cv2-mat_elem.out cv2-mat_elem.cpp -lopencv_core
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
  cv::Mat_<float> m1(3,4);
  // cv::Mat m1(3,3,CV_32F);
  m1= (cv::Mat_<float>(3,4)<<1,2,3,4, 5,6,7,8, 9,10,11,12);
  print(m1);
  print(m1.rows);
  print(m1.cols);

  print(m1.at<float>(0,2));
  print(m1.at<float>(cv::Point(0,2)));
  print(m1.at<float>(2,0));
  print(m1.at<float>(cv::Point(2,0)));

  return 0;
}
//-------------------------------------------------------------------------------------------
