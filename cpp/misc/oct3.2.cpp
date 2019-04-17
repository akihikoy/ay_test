//-------------------------------------------------------------------------------------------
/*! \file    oct3.2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jul.11, 2011
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
#include <octave/config.h>
#include <octave/Matrix.h>
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

int main(int argc, char**argv)
{
  ColumnVector x(4,1.0);
  print(x.transpose());
  print(x.dim1());
  print(x.dim2());
  print(x.length());
  ColumnVector y(x);
  print(y.transpose());
  print(y.dim1());
  print(y.dim2());
  print(y.length());
  ColumnVector z;
  z.resize(x.length(),2);
  print(z.transpose());
  print(z.dim1());
  print(z.dim2());
  print(z.length());
  // Matrix m;
  // // m.resize(3,3,1.0);
  // m= Matrix(DiagMatrix(z));
  // print(m.rows());
  // print(m.cols());
  // print(m.length());
  // CHOL chol(m);
  // Matrix l(chol.chol_matrix().transpose());
  // print(l.rows());
  // print(l.cols());
  // print(l.length());
  // ColumnVector a(l.length(),1.0);
  // print((l*a).transpose());
  return 0;
}
//-------------------------------------------------------------------------------------------
