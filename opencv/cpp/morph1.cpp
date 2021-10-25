//-------------------------------------------------------------------------------------------
/*! \file    morph1.cpp
    \brief   Test of morphing.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.21, 2018

g++ -g -Wall -O2 -o morph1.out morph1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

The idea is learned from:
http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.cpp
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cap_open.h"
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

void OnMouse(int event, int x, int y, int flags, void *vp_point)
{
  cv::Point &point(*reinterpret_cast<cv::Point*>(vp_point));

  if(event==cv::EVENT_LBUTTONDOWN || (event==cv::EVENT_MOUSEMOVE && flags&cv::EVENT_FLAG_LBUTTON))
  {
    point.x= x;
    point.y= y;
  }
  else if(event==cv::EVENT_RBUTTONDOWN)
  {
    point.x= 320;
    point.y= 240;
  }
}
//-------------------------------------------------------------------------------------------

// Warps triangular regions from src to dst
// morphTriangle from: https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.cpp
void MorphTriangle(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point2f> &t_src, std::vector<cv::Point2f> &t_dst)
{
  // Find bounding rectangle for each triangle
  cv::Rect r_dst= cv::boundingRect(t_dst);
  cv::Rect r_src= cv::boundingRect(t_src);

  // Offset points by left top corner of the respective rectangles
  std::vector<cv::Point2f> t0_src, t0_dst;
  std::vector<cv::Point> t0_dst_i;
  for(int i(0); i<3; ++i)
  {
    t0_dst.push_back(cv::Point2f(t_dst[i].x - r_dst.x, t_dst[i].y - r_dst.y));
    t0_dst_i.push_back(cv::Point(t_dst[i].x - r_dst.x, t_dst[i].y - r_dst.y)); // for fillConvexPoly
    t0_src.push_back(cv::Point2f(t_src[i].x - r_src.x, t_src[i].y -  r_src.y));
  }

  // Apply warpAffine to small rectangular patches
  cv::Mat src_r;
  src(r_src).copyTo(src_r);

  cv::Mat dst_r= cv::Mat::zeros(r_dst.height, r_dst.width, src_r.type());

  // Given a pair of triangles, find the affine transform.
  cv::Mat m_warp= cv::getAffineTransform(t0_src, t0_dst);
  // Apply the Affine Transform just found to the src_r image
  cv::warpAffine(src_r, dst_r, m_warp, dst_r.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

  // Get mask by filling triangle
  cv::Mat mask= cv::Mat::zeros(r_dst.height, r_dst.width, CV_8UC3);
  cv::fillConvexPoly(mask, t0_dst_i, cv::Scalar(1,1,1), 16, 0);

  // Copy triangular region of the rectangular patch to the output image
  cv::multiply(dst_r,mask, dst_r);
  cv::multiply(dst(r_dst), cv::Scalar(1,1,1) - mask, dst(r_dst));
  dst(r_dst) = dst(r_dst) + dst_r;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  std::vector<std::vector<cv::Point> >  polygon(1);

  cv::Point point_morph(320,240);

  // cv::namedWindow("camera",1);
  cv::namedWindow("morphed",1);
  cv::setMouseCallback("morphed", OnMouse, &point_morph);
  cv::Mat frame;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    // if(polygon[0].size()>0)
    // {
      // cv::fillPoly(frame, polygon, CV_RGB(128,0,128));
      // cv::polylines(frame, polygon, /*isClosed=*/true, CV_RGB(255,0,255), 2);
    // }

    cv::Mat morphed;
    frame.copyTo(morphed);
    std::vector<cv::Point2f> t_src, t_dst;
    t_src.clear();
    t_dst.clear();
    t_src.push_back(cv::Point2f(150,100));
    t_src.push_back(cv::Point2f(320,240));
    t_src.push_back(cv::Point2f(150,380));
    t_dst.push_back(cv::Point2f(150,100));
    t_dst.push_back(point_morph);
    t_dst.push_back(cv::Point2f(150,380));
    MorphTriangle(frame, morphed, t_src, t_dst);
    t_src.clear();
    t_dst.clear();
    t_src.push_back(cv::Point2f(150,100));
    t_src.push_back(cv::Point2f(320,240));
    t_src.push_back(cv::Point2f(490,100));
    t_dst.push_back(cv::Point2f(150,100));
    t_dst.push_back(point_morph);
    t_dst.push_back(cv::Point2f(490,100));
    MorphTriangle(frame, morphed, t_src, t_dst);
    t_src.clear();
    t_dst.clear();
    t_src.push_back(cv::Point2f(490,100));
    t_src.push_back(cv::Point2f(320,240));
    t_src.push_back(cv::Point2f(490,380));
    t_dst.push_back(cv::Point2f(490,100));
    t_dst.push_back(point_morph);
    t_dst.push_back(cv::Point2f(490,380));
    MorphTriangle(frame, morphed, t_src, t_dst);
    t_src.clear();
    t_dst.clear();
    t_src.push_back(cv::Point2f(490,380));
    t_src.push_back(cv::Point2f(320,240));
    t_src.push_back(cv::Point2f(150,380));
    t_dst.push_back(cv::Point2f(490,380));
    t_dst.push_back(point_morph);
    t_dst.push_back(cv::Point2f(150,380));
    MorphTriangle(frame, morphed, t_src, t_dst);

    // cv::imshow("camera", frame);
    cv::imshow("morphed", morphed);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
