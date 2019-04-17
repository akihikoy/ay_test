//-------------------------------------------------------------------------------------------
/*! \file    partial-spec.cpp
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
#include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

// #define FUNC
#define FUNC_OBJ

#ifdef FUNC
struct TTest
{
  template <typename T>
  void Func (const T &x);
};

// OK:
template<> void TTest::Func (const int &x)
{
  std::cout<<"x is "<<x<<std::endl;
}

// error: function template partial specialization ‘Func<std::vector<T2, std::allocator<_CharT> > >’ is not allowed
template<typename T2>
void TTest::Func<std::vector<T2> > (const std::vector<T2> &x)
{
  std::cout<<"x is";
  for (typename std::vector<T2>::const_iterator itr(x.begin()),last(x.end());itr!=last;++itr)
    std::cout<<" "<<*itr;
  std::cout<<std::endl;
}
#endif

#ifdef FUNC_OBJ
struct TTest
{
  template <typename T>
  void Func (const T &x)  {FuncObj<T>()(x);}

  template <typename T>
  struct FuncObj
  {
    void operator() (const T &x);
  };
};

// OK:
template<> struct TTest::FuncObj<int>
{
  void operator() (const int &x)
  {
    std::cout<<"x is "<<x<<std::endl;
  }
};

// error: function template partial specialization ‘Func<std::vector<T2, std::allocator<_CharT> > >’ is not allowed
template<typename T2>
struct TTest::FuncObj<std::vector<T2> >
{
  void operator() (const std::vector<T2> &x)
  {
    std::cout<<"x is";
    for (typename std::vector<T2>::const_iterator itr(x.begin()),last(x.end());itr!=last;++itr)
      std::cout<<" "<<*itr;
    std::cout<<std::endl;
  }
};
#endif


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
  TTest test;
  test.Func(12);
  vector<double> vec;
  vec.push_back(2.5); vec.push_back(25.1); vec.push_back(-0.1);
  test.Func(vec);
  return 0;
}
//-------------------------------------------------------------------------------------------
