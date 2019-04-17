//-------------------------------------------------------------------------------------------
/*! \file    spec-spec.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.06, 2010
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <iostream>
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
// #define print(var) print_container((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TTest
{
  double X;
  template <typename T>
  T Convert (void);
};

template<> int TTest::Convert (void)  // specialization
{
  return (int)X;
}

template <typename t_float>
t_float TTest::Convert (void)  // partial specialization
{
  return (t_float)(X*10.0l);
}

int main(int argc, char**argv)
{
  TTest t;
  t.X= 3.14;
  print(t.X);
  print(t.Convert<int>());
  print(t.Convert<float>());
  return 0;
}
//-------------------------------------------------------------------------------------------
