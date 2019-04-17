//-------------------------------------------------------------------------------------------
/*! \file    varmanip2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.22, 2010
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
#include <string>
// #include <vector>
#include <map>
#include <boost/bind.hpp>
#include <boost/function.hpp>
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
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

typedef std::string TIdentifier;

template <typename t_var>
struct get_composite_element_generator
{
  map<TIdentifier,int*>  m;
  get_composite_element_generator(t_var &x)
    {
      x.getmap(m);
    }
  int& operator()(t_var *x, const TIdentifier &rhs)
    {
      return *(m[rhs]);
    }
};

struct TTest
{
  int x;
  int y;
  int z;

  void getmap(map<TIdentifier,int*> &m)
    {
      m["x"]= &x;
      m["y"]= &y;
      m["z"]= &z;
    }
};

boost::function<int&(const TIdentifier &)>  Generate (TTest &var)
{
  return boost::bind(&get_composite_element_generator<TTest>::operator(),
                        get_composite_element_generator<TTest>(var),&var,_1);
}

int main(int argc, char**argv)
{
  // typedef double pt_real;
  // typedef int pt_int;

  TTest var;
  boost::function<int&(const TIdentifier &)> elem= Generate(var);

  elem("x")=10;
  elem("y")=-10;
  elem("z")=123;
  print(elem("x"));
  print(elem("y"));
  print(elem("z"));

  print(var.x);
  print(var.y);
  print(var.z);

  return 0;
}
//-------------------------------------------------------------------------------------------
