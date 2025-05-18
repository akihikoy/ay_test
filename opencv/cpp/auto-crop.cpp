//-------------------------------------------------------------------------------------------
/*! \file    auto-crop.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.4, 2015

    g++ -g -Wall -O2 -o auto-crop.out auto-crop.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
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

// Automatically crop the image whose background is black.
cv::Mat AutoCrop(const cv::Mat &src)
{
  cv::Mat gray, thresh;
  if(src.channels()==1)       src.copyTo(gray);
  else if(src.channels()==3)  cv::cvtColor(src,gray,cv::COLOR_BGR2GRAY);
  cv::threshold(gray,thresh,1,255,cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(thresh,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Point> &cnt(contours[0]);
  cv::Rect bound= cv::boundingRect(cnt);
  return src(bound);
}

int main(int argc, char**argv)
{
  cv::namedWindow("original",1);
  cv::namedWindow("cropped",1);

  cv::Mat src= cv::imread("sample/rtrace1.png");
  cv::imshow("original",src);

  cv::Mat crop= AutoCrop(src);
  cv::imshow("cropped",crop);
  cv::waitKey();

  return 0;
}
//-------------------------------------------------------------------------------------------
