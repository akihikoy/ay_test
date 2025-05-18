//-------------------------------------------------------------------------------------------
/*! \file    multi_tmpl_sbtr2.cpp
    \brief   Variable background subtraction with multiple templates.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.21, 2018

g++ -g -Wall -O2 -o multi_tmpl_sbtr2.out multi_tmpl_sbtr2.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

./multi_tmpl_sbtr2.out  "http://aypi11:8080/?action=stream?dummy=file.mjpg"
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
namespace cv
{

// For saving vector of vector.
template<typename T>
void write(cv::FileStorage &fs, const std::string&, const std::vector<std::vector<T> > &x)
{
  fs<<"[";
  for(typename std::vector<std::vector<T> >::const_iterator itr(x.begin()),end(x.end());itr!=end;++itr)
  {
    fs<<*itr;
  }
  fs<<"]";
}
//-------------------------------------------------------------------------------------------

}  // namespace cv
//-------------------------------------------------------------------------------------------
// Header
namespace loco_rabbits
{

/* Find an index idx of a sorted vector vec such that vec[idx]<=val<vec[idx+1].
  We can insert val by vec.insert(vec.begin()+idx+1, val) with keeping the sort.
  If vec.size()==0, idx=-1.
  If val<vec[0], idx=-1.
  If vec[vec.size()-1]<=val, idx=vec.size()-1. */
template<typename t_vector>
int FindIndex(const t_vector &vec, typename t_vector::const_reference val)
{
  for(int idx(0),end(vec.size()); idx<end; ++idx)
    if(val<vec[idx])  return idx-1;
  return vec.size()-1;
}
//-------------------------------------------------------------------------------------------

// cv::absdiff with mask: res=abs(a-b)
inline void absdiff(const cv::Mat &a, const cv::Mat &b, cv::Mat &res, cv::InputArray mask=cv::noArray(), int dtype=-1)
{
  cv::Mat aa,bb,cc;
  a.convertTo(aa, CV_16SC3);
  b.convertTo(bb, CV_16SC3);
  cv::subtract(aa, bb, cc, mask, dtype);
  cc= cv::abs(cc);
  cc.convertTo(res, a.type());
}
//-------------------------------------------------------------------------------------------

// Warps and alpha blends triangular regions from img1 and img2 to img.
void MorphBlendTriangles(cv::Mat &img1, cv::Mat &img2, cv::Mat &img, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2, std::vector<cv::Point2f> &t, double alpha);

//-------------------------------------------------------------------------------------------

// Template information to add a new template to TTemplateInterpolator.
struct TTITemplate
{
  cv::Mat Frame;
  std::vector<cv::Point> Polygon;
  double Position;
};
//-------------------------------------------------------------------------------------------

class TTemplateInterpolator
{
public:
  void Clear();

  // Find an index idx that satisfies position_[idx]<=pos<position_[idx+1].
  int FindInterval(const double &pos);

  // Save templates into file.
  void SaveToFile(const std::string &file_name);
  // Load templates from file.
  void LoadFromFile(const std::string &file_name);

  cv::Mat PolygonToMask(const cv::Size &size, const std::vector<cv::Point> &polygon);

  void AddTmpl(const TTITemplate &tmpl);

  /*
  Interpolate templates:
    i1=FindInterval(pos), i2=i1+1, alpha=(pos-position_[i1])/(position_[i2]-position_[i1])
    polygon_[i1], polygon_[i2], alpha --> polygon_i
    tmpl_img_[i1], tmpl_img_[i2], alpha --> tmpl_img_i
    corresponding mask of polygon_i --> mask_i
  Output variables: polygon_i,tmpl_img_i,mask_i,alpha
  Return: true if success, false if failed.
  */
  bool InterpolateTmpl(
      const double &pos,
      std::vector<cv::Point>  &polygon_i,
      cv::Mat &tmpl_img_i, cv::Mat &mask_i, double &alpha);

private:
  std::vector<std::vector<cv::Point> > polygon_;
  std::vector<cv::Mat> tmpl_img_;
  std::vector<double> position_;  // Sorted positions corresponding with other vectors.

};
//-------------------------------------------------------------------------------------------


}  // end of loco_rabbits (header)
//-------------------------------------------------------------------------------------------
// Implementation
namespace loco_rabbits
{

// Warps and alpha blends triangular regions from img1 and img2 to img.
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


//-------------------------------------------------------------------------------------------
// class TTemplateInterpolator
//-------------------------------------------------------------------------------------------

void TTemplateInterpolator::Clear()
{
  polygon_.clear();
  tmpl_img_.clear();
  position_.clear();
}
//-------------------------------------------------------------------------------------------

// Find an index idx that satisfies position_[idx]<=pos<position_[idx+1].
int TTemplateInterpolator::FindInterval(const double &pos)
{
  return FindIndex(position_, pos);
}
//-------------------------------------------------------------------------------------------

// Save templates into file.
void TTemplateInterpolator::SaveToFile(const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs<<"Polygon"<<polygon_;
  fs<<"Position"<<position_;
  fs<<"Templates"<<tmpl_img_;
  fs.release();
}
//-------------------------------------------------------------------------------------------

// Load templates from file.
void TTemplateInterpolator::LoadFromFile(const std::string &file_name)
{
  polygon_.clear();
  position_.clear();
  tmpl_img_.clear();

  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  fs["Polygon"]>>polygon_;
  fs["Position"]>>position_;
  fs["Templates"]>>tmpl_img_;
  fs.release();

  // for(int i(0),end(polygon_.size());i<end;++i)
  // {
    // for(int i2(0),end2(polygon_[i2].size());i2<end2;++i2)
      // std::cerr<<polygon_[i][i2]<<std::endl;
    // std::cerr<<std::endl;
  // }
}
//-------------------------------------------------------------------------------------------

cv::Mat TTemplateInterpolator::PolygonToMask(const cv::Size &size, const std::vector<cv::Point> &polygon)
{
  cv::Mat mask;
  mask.create(size, CV_8UC1);
  mask.setTo(0);
  std::vector<std::vector<cv::Point> >  polygon2;
  polygon2.push_back(polygon);
  cv::fillPoly(mask, polygon2, cv::Scalar(255));
  return mask;
}
//-------------------------------------------------------------------------------------------

void TTemplateInterpolator::AddTmpl(const TTITemplate &tmpl)
{
  int idx= FindInterval(tmpl.Position);

  polygon_.insert(polygon_.begin()+idx+1, tmpl.Polygon);
  position_.insert(position_.begin()+idx+1, tmpl.Position);

  cv::Mat mask= PolygonToMask(tmpl.Frame.size(), tmpl.Polygon);
  // mask.create(tmpl.Frame.size(), CV_8UC1);
  // mask.setTo(0);
  // std::vector<std::vector<cv::Point> >  polygon;
  // polygon.push_back(tmpl.Polygon);
  // cv::fillPoly(mask, polygon, cv::Scalar(255));

  cv::Mat tmpl_img;
  tmpl_img.create(tmpl.Frame.size(), tmpl.Frame.type());
  tmpl_img.setTo(0);
  tmpl.Frame.copyTo(tmpl_img, mask);

  tmpl_img_.insert(tmpl_img_.begin()+idx+1, tmpl_img);
}
//-------------------------------------------------------------------------------------------

/*
Interpolate templates:
  i1=FindInterval(pos), i2=i1+1, alpha=(pos-position_[i1])/(position_[i2]-position_[i1])
  polygon_[i1], polygon_[i2], alpha --> polygon_i
  tmpl_img_[i1], tmpl_img_[i2], alpha --> tmpl_img_i
  corresponding mask of polygon_i --> mask_i
Output variables: polygon_i,tmpl_img_i,mask_i,alpha
Return: true if success, false if failed.
*/
bool TTemplateInterpolator::InterpolateTmpl(
    const double &pos,
    std::vector<cv::Point>  &polygon_i,
    cv::Mat &tmpl_img_i, cv::Mat &mask_i, double &alpha)
{
  int i1=FindInterval(pos);
  int i2=i1+1;

  if(tmpl_img_.size()==0)  return false;
  if(tmpl_img_.size()==1 || i1==-1)
  {
    alpha= 0.0;
    polygon_i= polygon_[0];
    tmpl_img_i= tmpl_img_[0];
    mask_i= PolygonToMask(tmpl_img_[0].size(), polygon_i);
    return true;
  }
  if(i1==int(tmpl_img_.size()-1))
  {
    alpha= 0.0;
    polygon_i= polygon_[i1];
    tmpl_img_i= tmpl_img_[i1];
    mask_i= PolygonToMask(tmpl_img_[i1].size(), polygon_i);
    return true;
  }

  alpha= (pos-position_[i1])/(position_[i2]-position_[i1]);

  polygon_i.clear();
  assert(polygon_[i1].size()==polygon_[i2].size());
  cv::Point2f center1(0.0), center2(0.0), center_i(0.0);  // center of polygons.
  for(int i(0),end(polygon_[i1].size()); i<end; ++i)
  {
    double x= (1.0-alpha)*polygon_[i1][i].x + alpha*polygon_[i2][i].x;
    double y= (1.0-alpha)*polygon_[i1][i].y + alpha*polygon_[i2][i].y;
    polygon_i.push_back(cv::Point(x,y));
    center1.x+= polygon_[i1][i].x;
    center1.y+= polygon_[i1][i].y;
    center2.x+= polygon_[i2][i].x;
    center2.y+= polygon_[i2][i].y;
    center_i.x+= x;
    center_i.y+= y;
  }
  center1.x/= double(polygon_[i1].size());
  center1.y/= double(polygon_[i1].size());
  center2.x/= double(polygon_[i1].size());
  center2.y/= double(polygon_[i1].size());
  center_i.x/= double(polygon_[i1].size());
  center_i.y/= double(polygon_[i1].size());

  // mask_i.create(tmpl_img_[i1].size(), CV_8UC1);
  // mask_i.setTo(0);
  // cv::fillPoly(mask_i, polygon_i, cv::Scalar(255));
  mask_i= PolygonToMask(tmpl_img_[i1].size(), polygon_i);

  tmpl_img_i.create(tmpl_img_[i1].size(), tmpl_img_[i1].type());
  tmpl_img_i.setTo(0);
  int tri_vertices[][3]=
    {{-1,0,1},{-1,1,2},{-1,2,3},{-1,3,0}};  // index -1 denotes the center.

  for(int j(0),end(sizeof(tri_vertices)/sizeof(tri_vertices[0])); j<end; ++j)
  {
    std::vector<cv::Point2f> t1,t2,t;
    for(int d(0);d<3;++d)
    {
      int i= tri_vertices[j][d];
      if(i==-1)
      {
        t1.push_back(center1);
        t2.push_back(center2);
        t.push_back(center_i);
      }
      else
      {
        t1.push_back(cv::Point2f(polygon_[i1][i].x, polygon_[i1][i].y));
        t2.push_back(cv::Point2f(polygon_[i2][i].x, polygon_[i2][i].y));
        t.push_back(cv::Point2f(polygon_i[i].x, polygon_i[i].y));
      }
    }
    MorphBlendTriangles(tmpl_img_[i1], tmpl_img_[i2], tmpl_img_i, t1, t2, t, alpha);
  }
  return true;
}
//-------------------------------------------------------------------------------------------


}  // end of loco_rabbits (implementation)
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------


TTemplateInterpolator *PTemplateInterpolator(NULL);

void OnMouse(int event, int x, int y, int flags, void *vp_tmpl)
{
  TTITemplate  &tmpl(*reinterpret_cast<TTITemplate*>(vp_tmpl));

  if(event==cv::EVENT_LBUTTONUP)
  {
    tmpl.Polygon.push_back(cv::Point(x,y));
  }
  else if(event==cv::EVENT_RBUTTONUP)
  {
    PTemplateInterpolator->Clear();
  }

  if(tmpl.Polygon.size()>=4)
  {
    PTemplateInterpolator->AddTmpl(tmpl);
    tmpl.Polygon.clear();
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;


  TTITemplate tmpl;
  TTemplateInterpolator tinterpolator;
  PTemplateInterpolator= &tinterpolator;
  cv::namedWindow("camera",1);
  cv::setMouseCallback("camera", OnMouse, &tmpl);

  int iposition(30);
  cv::createTrackbar("position", "camera", &iposition, 100, NULL);

  cv::Mat frame;
  cv::Mat disp_img;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    tmpl.Frame= frame;
    tmpl.Position= double(iposition)/100.0;

    frame.copyTo(disp_img);
    if(tmpl.Polygon.size()>0)
    {
      std::vector<std::vector<cv::Point> >  polygon;
      polygon.push_back(tmpl.Polygon);
      // cv::fillPoly(disp_img, polygon, cv::Scalar(128,0,128));
      cv::polylines(disp_img, polygon, /*isClosed=*/true, cv::Scalar(255,0,255), 2);
    }


    std::vector<cv::Point>  polygon_i;
    cv::Mat tmpl_img_i, mask_i;
    double alpha;
    if(tinterpolator.InterpolateTmpl(tmpl.Position, polygon_i, tmpl_img_i, mask_i, alpha))
    {
      // cv::imshow("template1",tmpl.tmpl_img[0]);
      // cv::imshow("template2",tmpl.tmpl_img[1]);
      cv::imshow("template-blended",tmpl_img_i);

      cv::Mat diff;
      absdiff(frame, tmpl_img_i, diff, /*mask=*/mask_i/*, int dtype=-1*/);
      // diff+= cv::Scalar(128,128,128);
      // diff= cv::abs(diff);
      cv::imshow("diff", diff*5.0);

      cv::Mat diff_abs;
      cv::cvtColor(diff, diff_abs, cv::COLOR_BGR2GRAY);
      cv::Mat disp_img3[3];
      cv::split(disp_img, disp_img3);
      // disp_img3[0]+= 0.5*mask_i;
      disp_img3[2]+= 10.0*diff_abs;
      cv::merge(disp_img3,3,disp_img);
    }

    cv::imshow("camera", disp_img);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c=='s')
    {
      std::cerr<<"saving..."<<std::endl;
      tinterpolator.SaveToFile("/tmp/mts/mts.yaml.gz");
    }
    else if(c=='l')
    {
      std::cerr<<"loading..."<<std::endl;
      tinterpolator.LoadFromFile("/tmp/mts/mts.yaml.gz");
    }
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
