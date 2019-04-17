//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.01, 2016

g++ -g -Wall -O2 -o cv2-file_storage.out cv2-file_storage.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
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
  cv::FileStorage fs("data/ext_usbcam1_stereo.yaml", cv::FileStorage::READ);
  if(!fs.isOpened())
  {
    std::cerr<<"Failed to open file"<<std::endl;
    return -1;
  }

  cv::Mat D1, K1, D2, K2, K3;
  double x(1.234);
  K3= (cv::Mat_<unsigned char>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  // fs["D1"] >> D1;
  // fs["K1"] >> K1;
  // fs["D2"] >> D2;
  // fs["K2"] >> K2;
  #define PROC_VAR(x)  if(!fs[#x].empty())  fs[#x] >> x;
  PROC_VAR( D1 );
  PROC_VAR( K1 );
  PROC_VAR( D2 );
  PROC_VAR( K2 );
  PROC_VAR( K3 );
  PROC_VAR( x  );
  #undef PROC_VAR
  print(D1);
  print(K1);
  print(D2);
  print(K2);
  print(K3);
  print(x);
  return 0;
}
//-------------------------------------------------------------------------------------------
