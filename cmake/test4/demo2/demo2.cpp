//-------------------------------------------------------------------------------------------
/*! \file    demo2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.19, 2012
*/
//-------------------------------------------------------------------------------------------
#include <cv.h>
#include <highgui.h>
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

int main(int argc, char**argv)
{
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
  {
    return -1;
  }
  cv::namedWindow("camera",1);
  cv::Mat frame;
  while(true)
  {
    cap >> frame;
    cv::imshow("camera", frame);
    int c(cv::waitKey(30));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
