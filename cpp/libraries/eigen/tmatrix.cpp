//-------------------------------------------------------------------------------------------
/*! \file    tmatrix.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.20, 2013
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
#include <lora/eigen.h>
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
using namespace Eigen;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"=\n"<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  {
    TVAMatrix<double> m1(2,3,1.0),m2(3,2),m3;
    Eigen::MatrixXd e1(2,3);
    m1(1,1)= 0;
    print(MToEig(m1));
    e1<<1,2,3, 4,5,6;
    print(e1);
    MToEig(m1)= e1;
    print(MToEig(m1));
    MToEig(m2)= MToEig(m1).transpose();
    print(MToEig(m2));
    m3= EigToM(MToEig(m1)*MToEig(m1).transpose());
    print(MToEig(m3));
    const TVAMatrix<double> m4(m3);
    print(MToEig(m4));
  }
  cout<<"--------"<<endl;
  {
    TVAVector<double> m1(1.0,3),m2;
    Eigen::VectorXd e1(3);
    m1(1)= 0;
    print(VToEig(m1));
    e1<<1,2,3;
    print(e1);
    VToEig(m1)= e1;
    print(VToEig(m1));
    m2= EigToV(VToEig(m1)+VToEig(m1));
    print(VToEig(m2));
    m2= m2+m2;
    print(VToEig(m2));
    const TVAVector<double> m3(m2);
    print(VToEig(m3));
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
