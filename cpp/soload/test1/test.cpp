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

void invoke_method(void)
{
  void *dl_handle;
  typedef void (*TPrint)(void);
  TPrint func;
  char *error;

  dl_handle= dlopen("libmain.so", RTLD_LAZY);
  if (!dl_handle)
  {
    cerr<<"error: "<<dlerror()<<endl;
    return;
  }

  func= reinterpret_cast<TPrint>(dlsym(dl_handle, "Print"));
  error= dlerror();
  if (error != NULL)
  {
    cerr<<"error: "<<error<<endl;
    return;
  }

  func();

  dlclose (dl_handle);
  return;
}

int main (int argc, char *argv[])
{
  invoke_method();

  return 0;
}
//-------------------------------------------------------------------------------------------
