//-------------------------------------------------------------------------------------------
/*! \file    thinning1.cpp
    \brief   Test of thinning method.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.16, 2021

g++ -g -Wall -O2 -o thinning1.out thinning1.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

$ ./thinning1.out sample/binary1.png
$ ./thinning1.out sample/opencv-logo.png
$ ./thinning1.out sample/water_coins.jpg

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "thinning/thinning.hpp"
#include "thinning/thinning.cpp"
#include <cstdio>
#include <sys/time.h>  // gettimeofday
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
// using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  cv::Mat img_in= cv::imread(argv[1], cv::IMREAD_COLOR);

  float resize_factor(0.5);
  bool inv_img(false);
  cv::namedWindow("Input",1);
  CreateTrackbar<bool>("Invert", "Input", &inv_img, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_factor", "Input", &resize_factor, 0.1, 1.0, 0.1, &TrackbarPrintOnTrack);

  while(true)
  {
    cv::Mat img;
    if(resize_factor!=1.0)
      cv::resize(img_in, img, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
    else
      img= img_in;

    /// Threshold the input image
    cv::Mat img_grayscale, img_binary;
    cv::cvtColor(img, img_grayscale, cv::COLOR_BGR2GRAY);
    if(inv_img)
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    else
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    /// Apply thinning to get a skeleton
    cv::Mat img_thinning_ZS, img_thinning_GH;
    double t0= GetCurrentTime();
    cv::ximgproc::thinning(img_binary, img_thinning_ZS, cv::ximgproc::THINNING_ZHANGSUEN);
    double t1= GetCurrentTime();
    cv::ximgproc::thinning(img_binary, img_thinning_GH, cv::ximgproc::THINNING_GUOHALL);
    double t2= GetCurrentTime();

    std::cout<<"Computation time:"<<endl
      <<"  THINNING_ZHANGSUEN: "<<t1-t0<<" sec"<<endl
      <<"  THINNING_GUOHALL: "<<t2-t1<<" sec"<<endl;

    /// Visualize results
    cv::Mat result_ZS(img.rows, img.cols, CV_8UC3), result_GH(img.rows, img.cols, CV_8UC3);
    cv::Mat in[] = {img_thinning_ZS, img_thinning_ZS, img_thinning_ZS};
    cv::Mat in2[] = {img_thinning_GH, img_thinning_GH, img_thinning_GH};
    int from_to[] = {0,0, 1,1, 2,2};
    cv::mixChannels(in, 3, &result_ZS, 1, from_to, 3);
    cv::mixChannels(in2, 3, &result_GH, 1, from_to, 3);
    result_ZS= 0.5*img + result_ZS;
    result_GH= 0.5*img + result_GH;
    cv::imshow("Input", img_in);
    cv::imshow("Thinning ZHANGSUEN", result_ZS);
    cv::imshow("Thinning GUOHALL", result_GH);

    char c(cv::waitKey(500));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
