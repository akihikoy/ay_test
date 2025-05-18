//-------------------------------------------------------------------------------------------
/*! \file    version_check.cpp
    \brief   Test of checking OpenCV version.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.07, 2017

g++ -g -Wall -O2 -o version_check.out version_check.cpp -lopencv_core -I/usr/include/opencv4
g++ -g -Wall -O2 -o version_check.out version_check.cpp -lopencv_core -I$HOME/.local/include -L$HOME/.local/lib -Wl,-rpath=$HOME/.local/lib -I/usr/include/opencv4
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

// #define CV_VERSION_EPOCH    2
// #define CV_VERSION_MAJOR    4
// #define CV_VERSION_MINOR    8

int main(int argc, char**argv)
{
  cout << "OpenCV version : " << CV_VERSION << endl;
  cout << "Major version : " << CV_MAJOR_VERSION << endl;
  cout << "Minor version : " << CV_MINOR_VERSION << endl;
  cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

  #if CV_MAJOR_VERSION>2 || (CV_MAJOR_VERSION==2 && (CV_MINOR_VERSION>4 || (CV_MINOR_VERSION==4 && CV_SUBMINOR_VERSION>=13)))
    cout << "CV version >= 2.4.13" << endl;
  #else
    cout << "CV version < 2.4.13" << endl;
  #endif

  cv::Mat frame;
  frame.copyTo(frame);

  return 0;
}
//-------------------------------------------------------------------------------------------
