//-------------------------------------------------------------------------------------------
/*! \file    polygon_clip.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.01, 2016

g++ -g -Wall -O2 -o polygon_clip.out polygon_clip.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

qplot -x /tmp/polygon.dat w l lw 2 t '"original"' /tmp/clip_rect.dat w l lw 2 t '"clip"' /tmp/polygon_clipped.dat w l lw 4 t '"clipped"'
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

namespace ns_polygon_clip
{
#define TP cv::Point_<t_value>
#define ROW Row<t_value>
#define ADD Add<t_value>

template<typename t_value>
inline TP Row(const cv::Mat &points2d, int r)
{
  if(r<0)  r= points2d.rows+r;
  return TP(points2d.at<t_value>(r,0), points2d.at<t_value>(r,1));
}

template<typename t_value>
inline void Add(cv::Mat &points2d, const TP &p)
{
  cv::Mat v(p);
  v= v.t();
  points2d.push_back(v);
}

template<typename t_value>
inline int IsLeftOf(const TP &edge_1, const TP &edge_2, const TP &test)
{
  TP tmp1(edge_2.x - edge_1.x, edge_2.y - edge_1.y);
  TP tmp2(test.x - edge_2.x, test.y - edge_2.y);
  t_value x = (tmp1.x * tmp2.y) - (tmp1.y * tmp2.x);
  if(x < 0)  return 0;
  else if(x > 0)  return 1;
  else  return -1;  // Colinear points
}

template<typename t_value>
int IsClockwise(const cv::Mat &polygon)
{
  if(polygon.rows<3)  return -1;
  TP p0= ROW(polygon,0);
  TP p1= ROW(polygon,1);
  int isLeft(-1);
  for(int r(0),r_end(polygon.rows); r<r_end; ++r)
  {
    isLeft= IsLeftOf<t_value>(p0, p1, ROW(polygon,r));
    if(isLeft>=0)  // some of the points may be colinear.  That's ok as long as the overall is a polygon
      return isLeft==0 ? 1 : 0;
  }
  return -1;  // All the points in the polygon are colinear
}

template<typename t_value>
inline bool IsInside(const TP &cp1, const TP &cp2, const TP &p)
{
  return (cp2.x-cp1.x)*(p.y-cp1.y) > (cp2.y-cp1.y)*(p.x-cp1.x);
}

template<typename t_value>
inline TP ComputeIntersection(const TP &cp1, const TP &cp2, const TP &s, const TP &e)
{
  TP dc(cp1.x - cp2.x, cp1.y - cp2.y);
  TP dp(s.x - e.x, s.y - e.y);
  t_value n1= cp1.x * cp2.y - cp1.y * cp2.x;
  t_value n2= s.x * e.y - s.y * e.x;
  t_value n3= 1.0 / (dc.x * dp.y - dc.y * dp.x);
  return TP((n1*dp.x - n2*dc.x) * n3, (n1*dp.y - n2*dc.y) * n3);
}

template<typename t_value>
cv::Mat ClipPolygon_(const cv::Mat &polygon_subject_in, const cv::Mat &polygon_clip_in)
{
  cv::Mat polygon_empty(0,2,cv::DataType<t_value>::type);
  cv::Mat polygon_subject, polygon_clip;
  switch(IsClockwise<t_value>(polygon_subject_in))
  {
  case -1:
    std::cerr<<"polygon_subject: All the points are colinear"<<std::endl;
    return polygon_empty;
  case  0:  polygon_subject_in.copyTo(polygon_subject);  break;
  case +1:  cv::flip(polygon_subject_in, polygon_subject, 0);  break;
  }
//*DBG*/std::cerr<<"polygon_clip="<<polygon_clip_in<<std::endl;
//*DBG*/std::cerr<<"IsClockwise<t_value>(polygon_clip_in)="<<IsClockwise<t_value>(polygon_clip_in)<<std::endl;
  switch(IsClockwise<t_value>(polygon_clip_in))
  {
  case -1:
    std::cerr<<"polygon_clip: All the points are colinear"<<std::endl;
    return polygon_empty;
  case  0:  polygon_clip_in.copyTo(polygon_clip);  break;
  case +1:  cv::flip(polygon_clip_in, polygon_clip, 0);  break;
  }

  cv::Mat output_list= polygon_subject;
  TP cp1= ROW(polygon_clip,-1);
//*DBG*/std::cerr<<"polygon_clip="<<polygon_clip<<std::endl;

  for(int i_pc(0),i_pc_end(polygon_clip.rows); i_pc<i_pc_end; ++i_pc)
  {
    TP cp2= ROW(polygon_clip,i_pc);
    cv::Mat input_list= output_list;
    output_list= cv::Mat(0,2,cv::DataType<t_value>::type);
//*DBG*/std::cerr<<"in,out-list="<<input_list<<", "<<output_list<<std::endl;
    if(input_list.rows==0)  return polygon_empty;
    TP s= ROW(input_list,-1);
//*DBG*/std::cerr<<"  s="<<s<<std::endl;
//*DBG*/std::cerr<<"  cp1="<<cp1<<std::endl;
//*DBG*/std::cerr<<"  cp2="<<cp2<<std::endl;

    for(int i_in(0),i_in_end(input_list.rows); i_in<i_in_end; ++i_in)
    {
      TP e= ROW(input_list,i_in);
//*DBG*/std::cerr<<"  e="<<e<<", "<<IsInside<t_value>(cp1,cp2,e)<<", "<<IsInside<t_value>(cp1,cp2,s)<<std::endl;
      if(IsInside<t_value>(cp1,cp2,e))
      {
        if(!IsInside<t_value>(cp1,cp2,s))
          ADD(output_list, ComputeIntersection<t_value>(cp1, cp2, s, e));
        ADD(output_list, e);
      }
      else if(IsInside<t_value>(cp1,cp2,s))
        ADD(output_list, ComputeIntersection<t_value>(cp1, cp2, s, e));
      s= e;
    }
    cp1= cp2;
  }
  return output_list;
}
#undef TP
#undef ROW
#undef ADD
}  // ns_polygon_clip

cv::Mat ClipPolygon(const cv::Mat &polygon_subject, const cv::Mat &polygon_clip)
{
  using namespace ns_polygon_clip;
  assert(polygon_subject.type()==polygon_clip.type());
  assert(polygon_subject.cols==2);
  assert(polygon_clip.cols==2);
  switch(polygon_subject.type())
  {
  case CV_32F:  return ClipPolygon_<float>(polygon_subject, polygon_clip);
  case CV_64F:  return ClipPolygon_<double>(polygon_subject, polygon_clip);
  case CV_16S:  return ClipPolygon_<short>(polygon_subject, polygon_clip);
  case CV_32S:  return ClipPolygon_<int>(polygon_subject, polygon_clip);
  }
  throw;
}
//-------------------------------------------------------------------------------------------


}
//-------------------------------------------------------------------------------------------
#include <cmath>
#include <cstdlib>
#include <fstream>
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline unsigned Srand(void)
{
  unsigned seed ((unsigned)time(NULL));
  srand(seed);
  return seed;
}
inline double Rand (const double &max)
{
  return (max)*static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}
inline double Rand (const double &min, const double &max)
{
  return Rand(max - min) + min;
}

void SavePoly(const cv::Mat &points2d, const std::string &file_name)
{
  ofstream ofs(file_name.c_str());
  if(points2d.rows==0)  return;
  for(int r(0),r_end(points2d.rows); r<r_end; ++r)
    ofs<<points2d.at<float>(r,0)<<" "<<points2d.at<float>(r,1)<<endl;
  ofs<<points2d.at<float>(0,0)<<" "<<points2d.at<float>(0,1)<<endl;
}

int main(int argc, char**argv)
{
  using namespace ns_polygon_clip;
  Srand();
  cv::Mat polygon(0,2,CV_32F);
  int N= 100;
  for(int i(0);i<N;++i)
  {
    float th= (float(i)+Rand(-0.4,0.4))*2.0*M_PI/float(N);
    float r= Rand(1.0, 100.0);
    float x= r*cos(th);
    float y= r*sin(th);
    Add<float>(polygon, cv::Point_<float>(x,y));
  }
  float p1x,p1y, p2x,p2y;
  // p1x= Rand(1.0, 100.0); p1y= Rand(1.0, 100.0); p2x= Rand(1.0, 100.0); p2y= Rand(1.0, 100.0);
  p1x= Rand(-100.0, 100.0); p1y= Rand(-100, 100.0); p2x= Rand(-100, 100.0); p2y= Rand(-100, 100.0);
  cv::Mat rect= (cv::Mat_<float>(4,2)<<p1x,p1y, p2x,p1y, p2x,p2y, p1x,p2y);
  // std::cerr<<"polygon="<<polygon<<std::endl;
  // std::cerr<<"rect="<<rect<<std::endl;

  // cv::Mat polygon= (cv::Mat_<float>(4,2)<<-1.69021, -87.6474, -40.5151, -2.28697, -25.867, 44.6599, 29.6493, -3.34294);
  // cv::Mat rect= (cv::Mat_<float>(4,2)<<-88.5993, 13.5659, 57.7085, 13.5659, 57.7085, -55.5519, -88.5993, -55.5519);
  // cv::Mat polygon= (cv::Mat_<float>(4,2)<<-8.52197, -19.9693,-45.7384, 17.8315,24.4617, 78.9929,61.4862, 30.6966);
  // cv::Mat rect= (cv::Mat_<float>(4,2)<<-45.4064, -10.5343,2.6671, -10.5343,2.6671, -28.4372,-45.4064, -28.4372);
  // cv::Mat polygon= (cv::Mat_<float>(4,2)<<45.672222, -2.9928033,1.6948826, 8.3420715,-77.632523, 33.676598,-29.458261, -45.078251);
  // cv::Mat rect= (cv::Mat_<float>(4,2)<<-61.265263, -16.213083,-37.288738, -16.213083,-37.288738, 72.973518,-61.265263, 72.973518);

  cv::Mat polygon2= ClipPolygon(polygon, rect);

  SavePoly(polygon, "/tmp/polygon.dat");
  SavePoly(rect, "/tmp/clip_rect.dat");
  SavePoly(polygon2, "/tmp/polygon_clipped.dat");

  return 0;
}
//-------------------------------------------------------------------------------------------
