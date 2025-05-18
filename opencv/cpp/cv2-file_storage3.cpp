//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.27, 2018

g++ -g -Wall -O2 -o cv2-file_storage3.out cv2-file_storage3.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
namespace cv
{

// This worked with OpenCV 2.4, however in OpenCV 3.x,
// this is already defined in opencv2/core/core.hpp
// template<typename T>
// void write(cv::FileStorage &fs, const cv::String&, const std::vector<std::vector<T> > &x)
// {
//   fs<<"[";
//   for(typename std::vector<std::vector<T> >::const_iterator itr(x.begin()),end(x.end());itr!=end;++itr)
//   {
//     fs<<*itr;
//   }
//   fs<<"]";
// }
//-------------------------------------------------------------------------------------------

// It seems that read for std::vector<T> is problematic in OpenCV 3.x.
// So, we define an alternative.
template<typename T>
void vec_read(const FileNode& node, T &x, const T &x_default=T())
{
  read(node,x,x_default);
}
template<typename T>
void vec_read(const FileNode& node, std::vector<T> &x, const std::vector<T> &x_default=std::vector<T>())
{
  x.clear();
  for(FileNodeIterator itr(node.begin()),itr_end(node.end()); itr!=itr_end; ++itr)
  {
    T y;
    vec_read(*itr,y);
    x.push_back(y);
  }
}
//-------------------------------------------------------------------------------------------

// It seems that write for std::vector<T> is problematic in OpenCV 3.x.
// So, we define an alternative.
template<typename T>
void vec_write(FileStorage &fs, const String&, const T &x)
{
  write(fs,"",x);
}
template<typename T>
void vec_write(FileStorage &fs, const String&, const std::vector<T> &x)
{
  fs<<"[";
  for(typename std::vector<T >::const_iterator itr(x.begin()),end(x.end());itr!=end;++itr)
  {
    vec_write(fs,"",*itr);
  }
  fs<<"]";
}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
}  // namespace cv
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int src[]= {0,1,2,5,6,3,4};
  std::vector<int> vec1(src,src+sizeof(src)/sizeof(src[0]));

  {
    cv::FileStorage fs("/tmp/1.yaml", cv::FileStorage::WRITE);
    // fs<<"vec"<<vec1;
    fs<<"vec"; vec_write(fs,"vec",vec1);
    fs.release();
  }

  {
    vec1.clear();
    cv::FileStorage fs("/tmp/1.yaml", cv::FileStorage::READ);
    fs["vec"]>>vec1;
    fs.release();
  }

  for(int i(0),end(vec1.size());i<end;++i)
    std::cerr<<" "<<vec1[i];
  std::cerr<<std::endl;

  /////////////////////////////

  std::vector<cv::Point> vec2;
  vec2.push_back(cv::Point(10,20));
  vec2.push_back(cv::Point(100,200));
  vec2.push_back(cv::Point(30,20));

  {
    cv::FileStorage fs("/tmp/2.yaml", cv::FileStorage::WRITE);
    // fs<<"vec"<<vec2;
    fs<<"vec"; vec_write(fs,"vec",vec2);
    fs.release();
  }

  {
    vec2.clear();
    cv::FileStorage fs("/tmp/2.yaml", cv::FileStorage::READ);
    // fs["vec"]>>vec2;
    vec_read(fs["vec"],vec2);
    fs.release();
  }

  for(int i(0),end(vec2.size());i<end;++i)
    std::cerr<<" "<<vec2[i];
  std::cerr<<std::endl;

  /////////////////////////////

  std::vector<std::vector<cv::Point> > vec3(3);
  vec3[0].push_back(cv::Point(10,20));
  vec3[0].push_back(cv::Point(100,200));
  vec3[0].push_back(cv::Point(30,20));
  vec3[1].push_back(cv::Point(100,200));
  vec3[1].push_back(cv::Point(100,200));
  vec3[2].push_back(cv::Point(1,2));
  vec3[2].push_back(cv::Point(1,20));

  {
    cv::FileStorage fs("/tmp/3.yaml", cv::FileStorage::WRITE);
    // fs<<"vec"<<vec3;
    fs<<"vec"; vec_write(fs,"vec",vec3);
    fs.release();
  }

  {
    vec3.clear();
    cv::FileStorage fs("/tmp/3.yaml", cv::FileStorage::READ);
    // fs["vec"]>>vec3;
    vec_read(fs["vec"],vec3);
    fs.release();
  }

  std::cerr<<"vec3.size()= "<<vec3.size()<<std::endl;
  for(int i(0),end(vec3.size());i<end;++i)
  {
    for(int i2(0),end2(vec3[i].size());i2<end2;++i2)
      std::cerr<<" "<<vec3[i][i2];
    std::cerr<<std::endl;
  }

  /////////////////////////////

  std::vector<std::vector<std::vector<cv::Point> > > vec4;
  vec4.push_back(std::vector<std::vector<cv::Point> >(3));
  vec4[0][0].push_back(cv::Point(10,20));
  vec4[0][0].push_back(cv::Point(100,200));
  vec4[0][0].push_back(cv::Point(30,20));
  vec4[0][1].push_back(cv::Point(100,200));
  vec4[0][1].push_back(cv::Point(100,200));
  vec4[0][2].push_back(cv::Point(1,2));
  vec4[0][2].push_back(cv::Point(1,20));
  vec4.push_back(std::vector<std::vector<cv::Point> >(2));
  vec4[1][0].push_back(cv::Point(1,2));
  vec4[1][0].push_back(cv::Point(1,20));
  vec4[1][1].push_back(cv::Point(10,20));
  vec4[1][1].push_back(cv::Point(100,20));

  {
    cv::FileStorage fs("/tmp/4.yaml", cv::FileStorage::WRITE);
    // fs<<"vec"<<vec4;
    fs<<"vec"; vec_write(fs,"vec",vec4);
    fs.release();
  }

  {
    vec4.clear();
    cv::FileStorage fs("/tmp/4.yaml", cv::FileStorage::READ);
    // fs["vec"]>>vec4;
    vec_read(fs["vec"],vec4);
    fs.release();
  }

  std::cerr<<"vec4.size()= "<<vec4.size()<<std::endl;
  for(int i(0),end(vec4.size());i<end;++i)
  {
    std::cerr<<"-"<<std::endl;
    for(int i2(0),end2(vec4[i].size());i2<end2;++i2)
    {
      for(int i3(0),end3(vec4[i][i2].size());i3<end3;++i3)
        std::cerr<<" "<<vec4[i][i2][i3];
      std::cerr<<std::endl;
    }
    std::cerr<<std::endl;
  }

  /////////////////////////////

  {
    cv::FileStorage fs("/tmp/x.yaml", cv::FileStorage::WRITE);
    // fs<<"vec1"<<vec1;
    // fs<<"vec2"<<vec2;
    // fs<<"vec3"<<vec3;
    // fs<<"vec4"<<vec4;
    fs<<"vec1"; vec_write(fs,"vec",vec1);
    fs<<"vec2"; vec_write(fs,"vec",vec2);
    fs<<"vec3"; vec_write(fs,"vec",vec3);
    fs<<"vec4"; vec_write(fs,"vec",vec4);
    fs.release();
  }

  {
    vec1.clear();
    vec2.clear();
    vec3.clear();
    vec4.clear();
    cv::FileStorage fs("/tmp/x.yaml", cv::FileStorage::READ);
    // fs["vec1"]>>vec1;
    // fs["vec2"]>>vec2;
    // fs["vec3"]>>vec3;
    // fs["vec4"]>>vec4;
    vec_read(fs["vec1"],vec1);
    vec_read(fs["vec2"],vec2);
    vec_read(fs["vec3"],vec3);
    vec_read(fs["vec4"],vec4);
    fs.release();
  }
  std::cerr<<"=========="<<std::endl;
  for(int i(0),end(vec1.size());i<end;++i)
    std::cerr<<" "<<vec1[i];
  std::cerr<<std::endl;
  for(int i(0),end(vec2.size());i<end;++i)
    std::cerr<<" "<<vec2[i];
  std::cerr<<std::endl;
  std::cerr<<"vec3.size()= "<<vec3.size()<<std::endl;
  for(int i(0),end(vec3.size());i<end;++i)
  {
    for(int i2(0),end2(vec3[i].size());i2<end2;++i2)
      std::cerr<<" "<<vec3[i][i2];
    std::cerr<<std::endl;
  }
  std::cerr<<"vec4.size()= "<<vec4.size()<<std::endl;
  for(int i(0),end(vec4.size());i<end;++i)
  {
    std::cerr<<"-"<<std::endl;
    for(int i2(0),end2(vec4[i].size());i2<end2;++i2)
    {
      for(int i3(0),end3(vec4[i][i2].size());i3<end3;++i3)
        std::cerr<<" "<<vec4[i][i2][i3];
      std::cerr<<std::endl;
    }
    std::cerr<<std::endl;
  }

  /////////////////////////////

  return 0;
}
//-------------------------------------------------------------------------------------------
