//-------------------------------------------------------------------------------------------
/*! \file    template-constructor.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.25, 2010
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

class TTest
{
public:

  // NOTE: テンプレートは特殊化できる（ユーザが定義できる）が，オーバーロードはできない
  template <typename T>
  TTest(const T &i)
    {
      cout<<"TTest is initialized with "<<i<<endl;
    }
};

template<> TTest::TTest (const int &i)
{
  cout<<"TTest: string initialization: "<<i<<endl;
}


int main(int argc, char**argv)
{
  TTest x(3), y(3.14), z("hoge");
  return 0;
}
//-------------------------------------------------------------------------------------------
