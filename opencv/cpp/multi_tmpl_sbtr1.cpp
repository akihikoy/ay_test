//-------------------------------------------------------------------------------------------
/*! \file    multi_tmpl_sbtr1.cpp
    \brief   Template subtraction with multiple (2) templates.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.21, 2018

g++ -g -Wall -O2 -o multi_tmpl_sbtr1.out multi_tmpl_sbtr1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

./multi_tmpl_sbtr1.out  "http://aypi11:8080/?action=stream?dummy=file.mjpg"
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

// cv::Point Corner(const std::vector<cv::Point> &pts)
// {
  // if(pts.size()==0)  return cv::Point();
  // cv::Point c(pts[0]);
  // for(std::vector<cv::Point>::const_iterator itr(pts.begin()),end(pts.end());
      // itr!=end; ++itr)
  // {
    // if(c.x>itr->x)  c.x= itr->x;
    // if(c.y>itr->y)  c.y= itr->y;
  // }
  // return c;
// }
//-------------------------------------------------------------------------------------------

// Warps triangular region from src to dst.
void MorphTriangle(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point2f> &t_src, std::vector<cv::Point2f> &t_dst)
{
  cv::Rect r_dst= cv::boundingRect(t_dst);
  cv::Rect r_src= cv::boundingRect(t_src);

  std::vector<cv::Point2f> t0_src, t0_dst;
  std::vector<cv::Point> t0_dst_i;
  for(int i(0); i<3; ++i)
  {
    t0_dst.push_back(cv::Point2f(t_dst[i].x - r_dst.x, t_dst[i].y - r_dst.y));
    t0_dst_i.push_back(cv::Point(t_dst[i].x - r_dst.x, t_dst[i].y - r_dst.y)); // for fillConvexPoly
    t0_src.push_back(cv::Point2f(t_src[i].x - r_src.x, t_src[i].y -  r_src.y));
  }

  cv::Mat src_r;
  src(r_src).copyTo(src_r);

  cv::Mat dst_r= cv::Mat::zeros(r_dst.height, r_dst.width, src_r.type());

  cv::Mat m_warp= cv::getAffineTransform(t0_src, t0_dst);
  cv::warpAffine(src_r, dst_r, m_warp, dst_r.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

  cv::Mat mask= cv::Mat::zeros(r_dst.height, r_dst.width, CV_8UC3);
  cv::fillConvexPoly(mask, t0_dst_i, cv::Scalar(1,1,1), 16, 0);

  cv::multiply(dst_r,mask, dst_r);
  cv::multiply(dst(r_dst), cv::Scalar(1,1,1) - mask, dst(r_dst));
  dst(r_dst) = dst(r_dst) + dst_r;
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

class TTemplates
{
public:
  std::vector<std::vector<cv::Point> >  polygon;
  std::vector<cv::Mat> tmpl_img, mask;
  // TTemplates() : polygon(1) {}

  cv::Mat tmpl_img_i, mask_i;

  void CreateTmpl()
    {
      polygon.push_back(std::vector<cv::Point>());
      tmpl_img.push_back(cv::Mat());
      mask.push_back(cv::Mat());
    }

  void InterpolateTmpl(int t1, int t2, float alpha)
    {
    }

private:
};
//-------------------------------------------------------------------------------------------


cv::Mat frame;

void OnMouse(int event, int x, int y, int flags, void *vp_tmpl)
{
  TTemplates  &tmpl(*reinterpret_cast<TTemplates*>(vp_tmpl));

  if(event==cv::EVENT_LBUTTONUP)
  {
    if(tmpl.polygon.size()==0)  tmpl.CreateTmpl();

    tmpl.polygon.back().push_back(cv::Point(x,y));
  }

  if(tmpl.polygon.size()>0 && (tmpl.polygon.back().size()==4 || event==cv::EVENT_RBUTTONUP))
  {
    tmpl.mask.back().create(frame.size(), CV_8UC1);
    tmpl.mask.back().setTo(0);
    std::vector<std::vector<cv::Point> >  polygon;
    polygon.push_back(tmpl.polygon.back());
    cv::fillPoly(tmpl.mask.back(), polygon, cv::Scalar(255));

    tmpl.tmpl_img.back().create(frame.size(), frame.type());
    tmpl.tmpl_img.back().setTo(0);
    frame.copyTo(tmpl.tmpl_img.back(), tmpl.mask.back());
    // cv::imshow("template",tmpl.tmpl_img.back());

    tmpl.CreateTmpl();

    // std::cout<<" "<<Corner(tmpl.polygon[0])<<std::endl;
    // for(size_t i(0); i<tmpl.polygon[0].size(); ++i)
      // std::cout<<" "<<tmpl.polygon[0][i];
    // std::cout<<std::endl;
    // tmpl.polygon[0].clear();
  }
}
//-------------------------------------------------------------------------------------------

// absdiff with mask.
void absdiff(const cv::Mat &a, const cv::Mat &b, cv::Mat &res, cv::InputArray mask=cv::noArray(), int dtype=-1)
{
  cv::Mat aa,bb,cc;
  a.convertTo(aa, CV_16SC3);
  b.convertTo(bb, CV_16SC3);
  cv::subtract(aa, bb, cc, mask, dtype);
  cc= cv::abs(cc);
  cc.convertTo(res, a.type());
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;


  TTemplates tmpl;
  cv::namedWindow("camera",1);
  cv::setMouseCallback("camera", OnMouse, &tmpl);

  int mixrate(30);
  cv::createTrackbar("mixrate", "camera", &mixrate, 100, NULL);

  // cv::Mat frame;
  cv::Mat disp_img;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    frame.copyTo(disp_img);
    if(tmpl.polygon.size()>0 && tmpl.polygon.back().size()>0)
    {
      std::vector<std::vector<cv::Point> >  polygon;
      polygon.push_back(tmpl.polygon.back());
      cv::fillPoly(disp_img, polygon, CV_RGB(128,0,128));
      cv::polylines(disp_img, polygon, /*isClosed=*/true, CV_RGB(255,0,255), 2);
    }

    if(tmpl.tmpl_img.size()>1 && !tmpl.tmpl_img[1].empty())
    {
      double alpha= double(mixrate)/100.0;

      /*
      Interpolate templates:
        tmpl.polygon[0], tmpl.polygon[1], alpha --> polygon
        tmpl.tmpl_img[0], tmpl.tmpl_img[1], alpha --> tmpl_img
        corresponding mask of polygon --> mask
      */

      std::vector<std::vector<cv::Point> >  polygon(1);
      assert(tmpl.polygon[0].size()==tmpl.polygon[1].size());
      for(int i(0),end(tmpl.polygon[0].size()); i<end; ++i)
      {
        double x= (1.0-alpha)*tmpl.polygon[0][i].x + alpha*tmpl.polygon[1][i].x;
        double y= (1.0-alpha)*tmpl.polygon[0][i].y + alpha*tmpl.polygon[1][i].y;
        polygon[0].push_back(cv::Point(x,y));
      }

      cv::Mat mask;
      mask.create(frame.size(), CV_8UC1);
      mask.setTo(0);
      cv::fillPoly(mask, polygon, cv::Scalar(255));

      cv::Mat tmpl_img;
      tmpl_img.create(frame.size(), frame.type());
      tmpl_img.setTo(0);
      int tri_vertices[][3]= {{0,1,2},{2,3,0}};

      for(int j(0),end(sizeof(tri_vertices)/sizeof(tri_vertices[0])); j<end; ++j)
      {
        std::vector<cv::Point2f> t1,t2,t;
        for(int d(0);d<3;++d)
        {
          int i= tri_vertices[j][d];
          t1.push_back(cv::Point2f(tmpl.polygon[0][i].x, tmpl.polygon[0][i].y));
          t2.push_back(cv::Point2f(tmpl.polygon[1][i].x, tmpl.polygon[1][i].y));
          t.push_back(cv::Point2f(polygon[0][i].x, polygon[0][i].y));
        }
        MorphBlendTriangles(tmpl.tmpl_img[0], tmpl.tmpl_img[1], tmpl_img, t1, t2, t, alpha);
      }

      cv::imshow("template1",tmpl.tmpl_img[0]);
      cv::imshow("template2",tmpl.tmpl_img[1]);
      cv::imshow("template-blended",tmpl_img);

      cv::Mat diff;
      absdiff(frame, tmpl_img, diff, /*mask=*/mask/*, int dtype=-1*/);
      // diff+= CV_RGB(128,128,128);
      // diff= cv::abs(diff);
      cv::imshow("diff", diff*5.0);

      cv::Mat diff_abs;
      cv::cvtColor(diff, diff_abs, CV_BGR2GRAY);
      cv::Mat disp_img3[3];
      cv::split(disp_img, disp_img3);
      // disp_img3[0]+= 0.5*mask;
      disp_img3[2]+= 10.0*diff_abs;
      cv::merge(disp_img3,3,disp_img);
    }

    else if(tmpl.tmpl_img.size()>0 && !tmpl.tmpl_img[0].empty())
    {
      cv::Mat diff;
      absdiff(frame, tmpl.tmpl_img[0], diff, /*mask=*/tmpl.mask[0]/*, int dtype=-1*/);
      // diff+= CV_RGB(128,128,128);
      // diff= cv::abs(diff);
      cv::imshow("diff", diff*5.0);
    }

    cv::imshow("camera", disp_img);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
