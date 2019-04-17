//-------------------------------------------------------------------------------------------
/*! \file    cv-type-gen.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.30, 2012
*/
//-------------------------------------------------------------------------------------------
#include <lora/cv.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
#include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{



}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
#define cprint(var) PrintContainer((var), #var"= ")
#define itrprint(var) do{std::cout<<#var"= ";PrintContainer(CVBegin(var),CVEnd(var));std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat_<double> m1(3,3);
  m1= (cv::Mat_<double>(3,3)<<1,2,3, 4,5,6, 7,8,9);
  print(m1);
  itrprint(m1);

  cv::Vec<double,3> v1;
  v1= cv::Vec<double,3>(1,2,3);
  print(cv::Mat(v1));
  itrprint(v1);

  cv::Matx<double,3,3> m2(1,2,3, 4,5,6, 7,8,9);
  print(cv::Mat(m2));
  itrprint(m2);

  cv::Mat m3(3,3,CV_64F);
  m3= (cv::Mat_<double>(3,3)<<1,2,3, 4,5,6, 7,8,9);
  print(m3);
//   cprint(m3);
//   itrprint(m3);

  return 0;
}
//-------------------------------------------------------------------------------------------
