//-------------------------------------------------------------------------------------------
/*! \file    isnan_isinf.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.30, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cmath>  // isnan, isinf
#include <cfloat>  // NaN, Inf
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

int main(int argc, char**argv)
{
  print(std::isnan(NAN));
  print(std::isnan(INFINITY));
  print(std::isnan(0.0));

  print(std::isinf(NAN));
  print(std::isinf(INFINITY));
  print(std::isinf(0.0));

  return 0;
}
//-------------------------------------------------------------------------------------------
