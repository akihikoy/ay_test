//-------------------------------------------------------------------------------------------
/*! \file    cv2-vec_point.cpp
    \brief   cv::Vec vs cv::Point.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.06, 2022

g++ -g -Wall -O2 -o cv2-vec_point.out cv2-vec_point.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------

// Access functions to an element of a cv::Vec, cv::Point, or cv::Point3_ with the same interface.
template<typename value_type, int N>
inline value_type& VecElem(cv::Vec<value_type,N> &v, int idx)  {return v(idx);}
template<typename value_type, int N>
inline const value_type& VecElem(const cv::Vec<value_type,N> &v, int idx)  {return v(idx);}
template<typename value_type>
inline value_type& VecElem(cv::Point_<value_type> &v, int idx)
{
  switch(idx)
  {
    case 0:  return v.x;
    case 1:  return v.y;
  }
  throw;
}
template<typename value_type>
inline const value_type& VecElem(const cv::Point_<value_type> &v, int idx)
{
  switch(idx)
  {
    case 0:  return v.x;
    case 1:  return v.y;
  }
  throw;
}
template<typename value_type>
inline value_type& VecElem(cv::Point3_<value_type> &v, int idx)
{
  switch(idx)
  {
    case 0:  return v.x;
    case 1:  return v.y;
    case 2:  return v.z;
  }
  throw;
}
template<typename value_type>
inline const value_type& VecElem(const cv::Point3_<value_type> &v, int idx)
{
  switch(idx)
  {
    case 0:  return v.x;
    case 1:  return v.y;
    case 2:  return v.z;
  }
  throw;
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
#ifndef LIBRARY
#include <iostream>
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
int main(int argc, char**argv)
{
  cv::Vec2f v1(1.2,2.3);
  print(v1);
  print(VecElem(v1,0));
  print(VecElem(v1,1));

  cv::Vec3d v2(1.1,2.1,3.1);
  print(v2);
  print(VecElem(v2,1));
  VecElem(v2,1)*= 10;
  print(VecElem(v2,1));
  print(v2);

  cv::Point2f p1(2.1,3.2);
  print(p1);
  print(VecElem(p1,0));
  print(VecElem(p1,1));

  cv::Point3i p2(0,0,0);
  print(p2);
  print(VecElem(p2,0));
  VecElem(p2,0)= VecElem(v2,0);
  VecElem(p2,1)= VecElem(v2,1);
  VecElem(p2,2)= VecElem(v2,2);
  print(p2);
  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
