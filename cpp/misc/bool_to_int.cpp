//-------------------------------------------------------------------------------------------
/*! \file    bool_to_int.cpp
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

// ostream& operator<< (ostream &lhs, const bool &rhs)
// {
  // if(rhs)  lhs<<"true";
  // else lhs<<"false";
  // return lhs;
// }

int main(int argc, char**argv)
{
  print(static_cast<bool>(3));
  print(static_cast<bool>(1));
  print(static_cast<bool>(0));
  print(static_cast<bool>(-3));
  print(static_cast<int>(true));
  print(static_cast<int>(false));
  return 0;
}
//-------------------------------------------------------------------------------------------
