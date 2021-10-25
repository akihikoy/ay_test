//-------------------------------------------------------------------------------------------
/*! \file    draw_cross.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.14, 2015

    g++ -g -Wall -O2 -o draw_cross.out draw_cross.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

inline void DrawCrossOnCenter(cv::Mat &img, int size, const cv::Scalar &col, int thickness=1)
{
  int hsize(size/2);
  cv::line(img, cv::Point(img.cols/2-hsize,img.rows/2), cv::Point(img.cols/2+hsize,img.rows/2), col, thickness);
  cv::line(img, cv::Point(img.cols/2,img.rows/2-hsize), cv::Point(img.cols/2,img.rows/2+hsize), col, thickness);
}
//-------------------------------------------------------------------------------------------

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
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    DrawCrossOnCenter(frame, 40, cv::Scalar(255,255,255));

    cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
