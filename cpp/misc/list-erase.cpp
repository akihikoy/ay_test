//-------------------------------------------------------------------------------------------
/*! \file    list-erase.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jan.26, 2012
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
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
#include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
#include <list>
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
#define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define exec(exp) std::cout<<#exp<<": "; exp; std::cout<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  list<int> a;
  for(int i(1);i<=10;++i) {a.push_back(i);if(i==3||i==7)a.push_back(0);}
  print(a);
  exec(list<int>::iterator itr((++find(a.rbegin(),a.rend(),0)).base()));
  exec(PrintContainer(itr,a.end()));
  exec(a.erase(itr,a.end()));
  print(a);
  return 0;
}
//-------------------------------------------------------------------------------------------
