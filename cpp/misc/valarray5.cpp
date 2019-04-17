//-------------------------------------------------------------------------------------------
/*! \file    valarray5.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.25, 2013
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
#include <valarray>
#include <numeric>
// #include <string>
// #include <vector>
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
//-------------------------------------------------------------------------------------------

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

template <typename t_iterator>
void Print(t_iterator itr, t_iterator last)
{
  for(;itr!=last;++itr)
    cout<<" "<<*itr;
}
#define printv(var) do{std::cout<<#var"= "; Print(&var[0],&var[0]+var.size()); std::cout<<std::endl;}while(0)


int main(int argc, char**argv)
{
  double xa[]= {1.,2.,3.,4.,5.};
  valarray<double> xv1(xa,5), xv2(1,5);
  xv2/=10.0;
  printv(xv1);
  printv(xv2);
  print(inner_product(&xv1[0],&xv1[0]+xv1.size(),&xv2[0],0.0));
  return 0;
}
//-------------------------------------------------------------------------------------------
