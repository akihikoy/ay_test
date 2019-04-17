//-------------------------------------------------------------------------------------------
/*! \file    binary2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.28, 2010
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
// #include <vector>
// #include <boost/lexical_cast.hpp>
#include "binary-test.h"
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  if (argc==1)
  {
    list<TData> test;
    // for (int i(0); i<500000; ++i)
    {
      test.push_back(Data(2.236/3.0));
      test.push_back(Data(-3000000));
      test.push_back(Data(-1.28/3.0));
      // test.push_back(Data(string("hoge hoge")));
      test.push_back(Data(255));
      // test.push_back(Data(string("gegege")));
      test.push_back(Data(4.22e+50/3.0));
      // test.push_back(Data(string("xxx")));
    }
    // SaveAsBinary("test.dat",test);
    SaveAsBase64("test.dat",test);
    cerr<<"---------"<<endl;
    PrintDataList(test);
    cerr<<"---------"<<endl;
    cerr<<"saved to test.dat"<<endl;
    cerr<<"write "<<test.size()<<endl;
  }
  else
  {
    list<TData> test;
    // LoadFromBinary(argv[1],test);
    LoadFromBase64(argv[1],test);
    cerr<<"read "<<test.size()<<endl;
    cerr<<"loaded from "<<argv[1]<<endl;
    cerr<<"---------"<<endl;
    PrintDataList(test);
    cerr<<"---------"<<endl;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
