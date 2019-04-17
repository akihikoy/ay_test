//-------------------------------------------------------------------------------------------
/*! \file    test.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.25, 2012
*/
//-------------------------------------------------------------------------------------------
#include <dlfcn.h>
#include <iostream>
//-------------------------------------------------------------------------------------------
#include "test.h"
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

/*extern*/void (*ExportFunction)(void)(NULL);

void execute(void)
{
cerr<<"test1"<<endl;
  void *dl_handle= dlopen("lib/libmain.so", RTLD_LAZY);
  if (!dl_handle)
  {
    cerr<<"error: "<<dlerror()<<endl;
  }

cerr<<"test2"<<endl;
dlopen("lib/libmain.so", RTLD_LAZY);

  if(ExportFunction)
    ExportFunction();
  else
    cerr<<"ExportFunction is not assigned."<<endl;

  dlclose (dl_handle);
  return;
}

int main (int argc, char *argv[])
{
  execute();

  return 0;
}
//-------------------------------------------------------------------------------------------
