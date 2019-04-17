//-------------------------------------------------------------------------------------------
/*! \file    spec-as-prv.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.08, 2010
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

class TTest
{
public:

  template <typename T>
  void Print (const T &x)
    {
      std::cout<<"x is "<<x<<std::endl;
    }

private:

  // ERROR: explicit specialization in non-namespace scope ‘class loco_rabbits::TTest’
  // template <> void Print (const TTest &);

};

std::ostream& operator<< (std::ostream &lhs, const TTest &rhs)
{
  lhs<<"TTest is empty";
  return lhs;
}


// NOTE: if we do not implement this specialization, a linker error arises
// if someone tries to use it!
template <> void TTest::Print (const TTest &);

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TTest test;

  test.Print(2);
  test.Print(3.14);
  // test.Print(test);

  return 0;
}
//-------------------------------------------------------------------------------------------
