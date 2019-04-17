//-------------------------------------------------------------------------------------------
/*! \file    find_biggest_cnt.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.01, 2015

    g++ -g -Wall -O2 -o find_biggest_cnt.out find_biggest_cnt.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
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

// Find the largest contour and return info. bin_src should be a binary image.
// WARNING: bin_src is modified.
bool FindLargestContour(const cv::Mat &bin_src,
    double *area=NULL,
    cv::Point2d *center=NULL,
    cv::Rect *bound=NULL,
    std::vector<cv::Point> *contour=NULL)
{
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(bin_src,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
  if(contours.size()==0)  return false;
  double a(0.0),a_max(0.0), i_max(0);
  for(int i(0),i_end(contours.size()); i<i_end; ++i)
  {
    a= cv::contourArea(contours[i],false);
    if(a>a_max)  {a_max= a;  i_max= i;}
  }
  std::vector<cv::Point> &cnt(contours[i_max]);
  if(area!=NULL)
    *area= a_max;
  if(center!=NULL)
  {
    cv::Moments mu= cv::moments(cnt);
    *center= cv::Point2d(mu.m10/mu.m00, mu.m01/mu.m00);
  }
  if(bound!=NULL)
    *bound= cv::boundingRect(cnt);
  if(contour!=NULL)
    *contour= cnt;
  return true;
}

int main(int argc, char**argv)
{
  cv::Mat src= cv::imread("sample/binary1.png");
  cv::Mat gray, mask;
  if(src.channels()==1)       src.copyTo(gray);
  else if(src.channels()==3)  cv::cvtColor(src,gray,cv::COLOR_BGR2GRAY);
  cv::threshold(gray,mask,1,255,cv::THRESH_BINARY);


  double area(0.0);
  cv::Point2d center;
  cv::Rect bound;
  std::vector<cv::Point> contour;
  FindLargestContour(mask, /*area=*/&area, /*center=*/&center, /*bound=*/&bound, /*contour=*/&contour);

  std::vector<std::vector<cv::Point> > contours;
  contours.push_back(contour);
  std::cerr<<"area="<<area<<std::endl;
  cv::drawContours(src, contours, 0, CV_RGB(0,255,0), /*thickness=*/2, /*linetype=*/8);
  cv::rectangle(src, bound, cv::Scalar(0,128,255), 2);
  cv::circle(src, center, 5, cv::Scalar(255,0,128));

  cv::namedWindow("image",1);
  cv::namedWindow("mask",1);
  cv::imshow("image",src);
  cv::imshow("mask",mask);
  cv::waitKey();

  return 0;
}
//-------------------------------------------------------------------------------------------
