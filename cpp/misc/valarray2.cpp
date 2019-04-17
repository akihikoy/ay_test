//-------------------------------------------------------------------------------------------
/*! \file    valarray2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.18, 2012
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
#include <valarray>
// #include <list>
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
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define printv(var) do{std::cout<<#var"= ";PrintContainer(&(var)[0],&(var)[(var).size()]);std::cout<<std::endl;}while(0)
#define printa(var) do{std::cout<<#var"= ";PrintContainer(&(var)[0],&(var)[SIZE_OF_ARRAY(var)]);std::cout<<std::endl;}while(0)
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  double xa[]= {1.,2.,3.,4.,5.};
  valarray<double> xv(xa,SIZE_OF_ARRAY(xa));
  int sa[]= {0,2,4};
  valarray<int> sv(sa,SIZE_OF_ARRAY(sa));

  printv(xv);
  printv(sv);
  return 0;
}
//-------------------------------------------------------------------------------------------
