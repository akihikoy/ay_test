//-------------------------------------------------------------------------------------------
/*! \file    boost-path3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.18, 2014
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
#include <boost/filesystem.hpp>
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
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline boost::filesystem::path ConcatenatePath(const boost::filesystem::path &lhs, const std::string &rhs)
{
  return boost::filesystem::path(lhs.string()+rhs);
}
//-------------------------------------------------------------------------------------------

/* We define this operator+ because in an older version (v.2),
    this function was defined, but it was removed in v.3. */
# if defined(BOOST_FILESYSTEM_VERSION) && BOOST_FILESYSTEM_VERSION >= 3
inline boost::filesystem::path operator+(const boost::filesystem::path &lhs, const std::string &rhs)
{
  return ConcatenatePath(lhs,rhs);
}
#endif
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  using namespace boost::filesystem;
  path A[5]={"prg/testl/a.out","prg/testl/test","prg","prg.out","prg/test/"};
  for(int i(0);i<5;++i)
  {
    cout<<"--------"<<endl;
    path &a(A[i]);
    print(a);
    print(a.filename());
    print(a.extension());
    // print(a.replace_extension(a.extension().string()+".old"));
    // print(a.replace_extension(a.extension().string()+"2"));
    print(path(a.string()+".old"));
    print(path(a.string()+"2"));
    print(a+".old");
    print(a+"2");
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
