//-------------------------------------------------------------------------------------------
/*! \file    oct-mat-seq.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.21, 2010
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
#include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
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
// #define print(var) print_container((var), #var"= ")
#define print(var) std::cout<<#var"= "<<std::endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  Matrix m= OctGen2<Matrix>(3,4, 1.,2.,3.,4., 5.,6.,7.,8., 9.,0.,1.,2.);
  print(m);
  cout<<"m= ";
  for(double *itr(OctBegin(m)),*last(OctEnd(m)); itr!=last; ++itr)
    cout<<" "<<*itr;
  cout<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
