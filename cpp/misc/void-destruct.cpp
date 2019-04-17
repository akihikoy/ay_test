//-------------------------------------------------------------------------------------------
/*! \file    void-destruct.cpp
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

struct TTest
{
  TTest()
    {
      std::cout<<"TTest is constructed"<<std::endl;
    }
  ~TTest()
    {
      std::cout<<"TTest is destructed"<<std::endl;
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

int main(int argc, char**argv)
{
#if 0
  TTest *test= new TTest();  // TTest::TTest() is called
  delete test;               // TTest::~TTest() IS called
#endif

#if 0
  void *test= new TTest();  // TTest::TTest() is called
  delete test;              // TTest::~TTest() is NOT called
#endif

#if 1
  void *test= new TTest();  // TTest::TTest() is called
  delete static_cast<TTest*>(test);  // TTest::~TTest() IS called
#endif

  return 0;
}
//-------------------------------------------------------------------------------------------
