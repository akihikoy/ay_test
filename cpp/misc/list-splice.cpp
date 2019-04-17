//-------------------------------------------------------------------------------------------
/*! \file    list-splice.cpp
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
  list<int> a,b;
  for(int i(1);i<=10;++i) {a.push_back(i);if(i==3||i==7)a.push_back(0);}
  for(int i(100);i<=101;++i) {b.push_back(i);}
  print(a);
  print(b);
  cout<<"----"<<endl;
  exec(PrintContainer(find(a.begin(),a.end(),0),a.end()));
  exec(PrintContainer(find(a.rbegin(),a.rend(),0),a.rend()));
  exec(PrintContainer(find(a.rbegin(),a.rend(),0).base(),a.end()));
  exec(b.splice(b.end(),a,find(a.rbegin(),a.rend(),0).base(),a.end()));
  print(a);
  print(b);
  return 0;
}
//-------------------------------------------------------------------------------------------
