//-------------------------------------------------------------------------------------------
/*! \file    morph2.cpp
    \brief   Test of morphing.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.26, 2018

g++ -g -Wall -O2 -o morph2.out morph2.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

The idea is learned from:
http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.cpp
*/
//-------------------------------------------------------------------------------------------
// #include <lora/small_classes.h>
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

size_t Idx(0);
void OnMouse(int event, int x, int y, int flags, void *vp_points)
{
  std::vector<cv::Point> &points(*reinterpret_cast<std::vector<cv::Point>*>(vp_points));

  if(event==cv::EVENT_LBUTTONDOWN)
  {
    std::cout<<"Changing point "<<Idx<<std::endl;
    points[Idx].x= x;
    points[Idx].y= y;
    ++Idx;
    if(Idx>=points.size())  Idx= 0;
  }
  else if(event==cv::EVENT_RBUTTONDOWN)
  {
    Idx= 0;
  }
}
//-------------------------------------------------------------------------------------------

// Warps and alpha blends triangular regions from img1 and img2 to img
// Code from: https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.cpp
void MorphBlendTriangles(cv::Mat &img1, cv::Mat &img2, cv::Mat &img, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2, std::vector<cv::Point2f> &t, double alpha)
{
  // Find bounding rectangle for each triangle
  cv::Rect r = cv::boundingRect(t);
  cv::Rect r1 = cv::boundingRect(t1);
  cv::Rect r2 = cv::boundingRect(t2);

  // Offset points by left top corner of the respective rectangles
  std::vector<cv::Point2f> t1Rect, t2Rect, tRect;
  std::vector<cv::Point> tRectInt;
  for(int i = 0; i < 3; i++)
  {
    tRect.push_back( cv::Point2f( t[i].x - r.x, t[i].y -  r.y) );
    tRectInt.push_back( cv::Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly

    t1Rect.push_back( cv::Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
    t2Rect.push_back( cv::Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
  }

  // Get mask by filling triangle
  cv::Mat mask = cv::Mat::zeros(r.height, r.width, img1.type());
  cv::fillConvexPoly(mask, tRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

  // Apply warpImage to small rectangular patches
  cv::Mat img1Rect, img2Rect;
  img1(r1).copyTo(img1Rect);
  img2(r2).copyTo(img2Rect);

  cv::Mat warpImage1 = cv::Mat::zeros(r.height, r.width, img1Rect.type());
  cv::Mat warpImage2 = cv::Mat::zeros(r.height, r.width, img2Rect.type());

  cv::Mat warpMat1 = cv::getAffineTransform( t1Rect, tRect );
  cv::warpAffine( img1Rect, warpImage1, warpMat1, warpImage1.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  cv::Mat warpMat2 = cv::getAffineTransform( t2Rect, tRect );
  cv::warpAffine( img2Rect, warpImage2, warpMat2, warpImage2.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

  // Alpha blend rectangular patches
  cv::Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

  // Copy triangular region of the rectangular patch to the output image
  cv::multiply(imgRect,mask, imgRect);
  cv::multiply(img(r), cv::Scalar(1.0,1.0,1.0) - mask, img(r));
  img(r) = img(r) + imgRect;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  std::vector<std::vector<cv::Point> >  polygon(1);

  // first 3 points: triangle-1, last 3 points: triangle-2.
  std::vector<cv::Point> points_morph;
  points_morph.push_back(cv::Point(160,100));
  points_morph.push_back(cv::Point(80,200));
  points_morph.push_back(cv::Point(240,200));
  points_morph.push_back(cv::Point(480,100));
  points_morph.push_back(cv::Point(400,200));
  points_morph.push_back(cv::Point(560,200));

  // cv::namedWindow("camera",1);
  cv::namedWindow("morphed",1);
  cv::setMouseCallback("morphed", OnMouse, &points_morph);

  int mixrate(50);
  cv::createTrackbar("mixrate", "morphed", &mixrate, 100, NULL);

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

    double alpha= double(mixrate)/100.0;

    cv::Mat morphed;
    frame.copyTo(morphed);
    std::vector<cv::Point2f> t1, t2, t;
    t1.push_back(points_morph[0]);
    t1.push_back(points_morph[1]);
    t1.push_back(points_morph[2]);
    t2.push_back(points_morph[3]);
    t2.push_back(points_morph[4]);
    t2.push_back(points_morph[5]);
    t.push_back(cv::Point2f((1.0-alpha)*points_morph[0].x+alpha*points_morph[3].x,
                            (1.0-alpha)*points_morph[0].y+alpha*points_morph[3].y));
    t.push_back(cv::Point2f((1.0-alpha)*points_morph[1].x+alpha*points_morph[4].x,
                            (1.0-alpha)*points_morph[1].y+alpha*points_morph[4].y));
    t.push_back(cv::Point2f((1.0-alpha)*points_morph[2].x+alpha*points_morph[5].x,
                            (1.0-alpha)*points_morph[2].y+alpha*points_morph[5].y));
    MorphBlendTriangles(frame, frame, morphed, t1, t2, t, alpha);

    // visualize polygons:
    std::vector<std::vector<cv::Point> >  p1(1),p2(1),p(1);
    p1[0].push_back(points_morph[0]);
    p1[0].push_back(points_morph[1]);
    p1[0].push_back(points_morph[2]);
    p2[0].push_back(points_morph[3]);
    p2[0].push_back(points_morph[4]);
    p2[0].push_back(points_morph[5]);
    p[0].push_back(t[0]);
    p[0].push_back(t[1]);
    p[0].push_back(t[2]);
    cv::polylines(morphed, p1, /*isClosed=*/true, CV_RGB(255,0,255), 2);
    cv::polylines(morphed, p2, /*isClosed=*/true, CV_RGB(255,0,255), 2);
    cv::polylines(morphed, p, /*isClosed=*/true, CV_RGB(255,0,255), 2);

    // cv::imshow("camera", frame);
    cv::imshow("morphed", morphed);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
