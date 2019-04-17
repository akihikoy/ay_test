//-------------------------------------------------------------------------------------------
/*! \file    stereo_flow.cpp
    \brief   Matching calculation for rectified stereo images.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.06, 2016

g++ -g -Wall -O2 -o stereo_flow.out stereo_flow.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

int main(int argc, char**argv)
{
  cv::VideoCapture vin1("sample/vout2l.avi");
  cv::VideoCapture vin2("sample/vout2r.avi");
  if(!vin1.isOpened() || !vin2.isOpened())
  {
    std::cerr<<"failed to open!"<<std::endl;
    return -1;
  }
  std::cerr<<"video files opened"<<std::endl;

  cv::namedWindow("video1",1);
  cv::namedWindow("video2",1);
  cv::namedWindow("stereo_flow",1);
  cv::Mat frame1, frame2;

  int n_disp(16*8), w_size(5);
  cv::StereoBM stereo(cv::StereoBM::BASIC_PRESET, /*ndisparities=*/n_disp, /*SADWindowSize=*/w_size);

  for(;;)
  {
    bool res1(vin1.read(frame1)), res2(vin2.read(frame2));
    if(!res1 || !res2)
    {
      std::cerr<<"video reached the end (looped)"<<std::endl;
      vin1.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
      vin2.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
      continue;
    }

    int we(3),wd(8);
    cv::erode(frame1,frame1,cv::Mat(),cv::Point(-1,-1), we);
    cv::dilate(frame1,frame1,cv::Mat(),cv::Point(-1,-1), wd);
    cv::erode(frame2,frame2,cv::Mat(),cv::Point(-1,-1), we);
    cv::dilate(frame2,frame2,cv::Mat(),cv::Point(-1,-1), wd);

    cv::Mat disparity;
    cv::cvtColor(frame1, frame1, CV_BGR2GRAY);
    cv::cvtColor(frame2, frame2, CV_BGR2GRAY);
    stereo(frame1, frame2, disparity, /*disptype=*/CV_32FC1);
    // /*TEST*/cv::erode(disparity,disparity,cv::Mat(),cv::Point(-1,-1), 2);
    cv::normalize(disparity, disparity, 0, 255, CV_MINMAX, CV_8U);
    cv::imshow("stereo_flow", disparity);

    cv::imshow("video1", frame1);
    cv::imshow("video2", frame2);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
