//-------------------------------------------------------------------------------------------
/*! \file    map-map.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jun.22, 2010
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
#include <lora/variable_space.h>
#include <lora/variable_space_impl.h>
#include <lora/variable_parser.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
#include <map>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
using namespace var_space;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  string filename("test.var");
  if(argc>1)  filename=argv[1];

  map<string, map<int,int> >  test;
  TVariable var_test(VariableSpace());
  var_test.AddMemberVariable("test",TVariable(test));
  bool is_last;
  if(!ParseFile(var_test,filename,&is_last))
    LERROR("fatal!");
  if(is_last) LMESSAGE("whole file has been parsed.");
  else        LMESSAGE("un-parsed lines are remaining.");
  var_test.WriteToStream(cout,true);
  return 0;
}
//-------------------------------------------------------------------------------------------
