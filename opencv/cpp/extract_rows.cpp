//-------------------------------------------------------------------------------------------
/*! \file    extract_rows.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.03, 2016

g++ -g -Wall -O2 -o extract_rows.out extract_rows.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
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
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

// Extract rows of src and store to dst (works for &dst==&src)
void ExtractRows(const cv::Mat &src, const std::vector<int> &idx, cv::Mat &dst)
{
  cv::Mat buf(src);
  dst.create(idx.size(),buf.cols,buf.type());
  int r(0);
  for(std::vector<int>::const_iterator itr(idx.begin()),itr_end(idx.end()); itr!=itr_end; ++itr,++r)
    buf.row(*itr).copyTo(dst.row(r));
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat m1(3,3,CV_32F), m2;
  m1= (cv::Mat_<float>(3,3)<<1.5,0,1, 0,0,1, 0,1,1);
  print(m1);
  std::vector<int> idx(2);
  idx[0]=0;idx[1]=2;
  ExtractRows(m1,idx,m2);
  print(m1);
  print(m2);
  idx[0]=1;idx[1]=0;
  ExtractRows(m1,idx,m1);
  print(m1);
  print(m2);
  return 0;
}
//-------------------------------------------------------------------------------------------
