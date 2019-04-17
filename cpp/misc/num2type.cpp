//-------------------------------------------------------------------------------------------
/*! \file    num2type.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.30, 2012
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

template <int id> struct IdToType {typedef void value_type;};

template <> struct IdToType<1> {typedef int value_type;};
template <> struct IdToType<2> {typedef double value_type;};

struct TTest
{
  int Id;
  long double Value;
};

template <typename t_x>
t_x Square(const TTest &t, t_x dummy/*=IdToType<t.Id>::value_type*/)
{
  return t.Value*t.Value;
}


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
  TTest t1;
  t1.Id=1;
  t1.Value= 2.5;
  print(Square(t1,IdToType</*t1.Id*/1>::value_type(0)));
  print(Square(t1,IdToType<2>::value_type(0)));
  return 0;
}
//-------------------------------------------------------------------------------------------
