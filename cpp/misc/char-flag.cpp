//-------------------------------------------------------------------------------------------
/*! \file    char-flag.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.17, 2010
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
static const char FA  (0x01);
static const char FB  (0x02);
static const char FC  (0x04);
static const char FD  (0x08);
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void check (char flags)
{
  #define DISP(x_f) if(x_f & flags)  cout<<" "#x_f;
  DISP(FA)
  DISP(FB)
  DISP(FC)
  DISP(FD)
  cout<<endl;
  #undef DISP
}

#define CHECK(x_fs)  {cout<<#x_fs"="; check(x_fs);}
int main(int argc, char**argv)
{
  CHECK(FA);
  CHECK(0);
  CHECK(FA|FD);
  CHECK(FB|FA|FC);
  CHECK(FD|FA|FB|FC);
  CHECK(FD);
  return 0;
}
//-------------------------------------------------------------------------------------------
