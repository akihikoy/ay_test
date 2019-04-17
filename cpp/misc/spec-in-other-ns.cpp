//-------------------------------------------------------------------------------------------
/*! \file    spec-in-other-ns.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.07, 2010
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

struct TTest
{
  double X;
  template <typename T>
  void PrintAs ()
    {
      std::cout<<static_cast<T>(X)<<std::endl;
    }
};

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

namespace loco_rabbits {
  // NOTE: if comment out the "namespace loco_rabbits", an error arises:
  // error: specialization of ‘template<class T> void loco_rabbits::TTest::PrintAs()’
  // in different namespace

template<> void ::loco_rabbits::TTest::PrintAs <int> ()
{
  std::cout<<"int(X) is "<<static_cast<int>(X)<<std::endl;
}

}

int main(int argc, char**argv)
{
  TTest t;
  t.X= 3.14;
  t.PrintAs<int>();
  t.PrintAs<double>();
  return 0;
}
//-------------------------------------------------------------------------------------------
