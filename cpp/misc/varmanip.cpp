//-------------------------------------------------------------------------------------------
/*! \file    varmanip.cpp
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
// #include <string>
// #include <vector>
// #include <list>
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


template <typename t_to, typename t_from>
inline t_to var_space_cast (const t_from &from)
{
  return static_cast<t_to>(from);
}
//-------------------------------------------------------------------------------------------

// TODO: specialize

template <typename t_from, typename t_to>
void converter_generator_pr(const t_from *from, t_to &to)
{
  to= var_space_cast<t_to>(*from);
}
template <typename t_from, typename t_to>
void converter_generator_rp(const t_from &from, t_to *to)
{
  *to= var_space_cast<t_to>(from);
}
//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  typedef double pt_real;
  typedef int pt_int;
  pt_real var;
  boost::function<void(pt_int&)>   get_as_int  = boost::bind(converter_generator_pr<pt_real  ,pt_int   >,&var,_1);
  boost::function<void(pt_real&)>  get_as_real = boost::bind(converter_generator_pr<pt_real  ,pt_real  >,&var,_1);
  boost::function<void(const pt_int&)>   set_by_int  = boost::bind(converter_generator_rp<pt_int   ,pt_real  >,_1,&var);
  boost::function<void(const pt_real&)>  set_by_real = boost::bind(converter_generator_rp<pt_real  ,pt_real  >,_1,&var);

  pt_int xi;
  pt_real xr;

  set_by_real(3.14);
  print(var);
  get_as_real(xr);  print(xr);
  get_as_int(xi);  print(xi);
  set_by_int(-2);
  print(var);
  get_as_real(xr);  print(xr);
  get_as_int(xi);  print(xi);

  return 0;
}
//-------------------------------------------------------------------------------------------
