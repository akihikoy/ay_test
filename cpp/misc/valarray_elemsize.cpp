//-------------------------------------------------------------------------------------------
/*! \file    valarray_elemsize.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.10, 2015
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <valarray>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

class TTest
{
public:
  TTest() : elem_(10)  {}
private:
  std::valarray<double> elem_;
};

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
  double xa[5]= {1.,2.,3.,4.,5.};
  valarray<double> xv(xa,5);
  print(sizeof(double));
  print(sizeof(xv[0]));
  print(&xv[0]);
  print(&xv[4]);
  print(&xv[0]+4);
  print(long(&xv[4]));
  print(long(&xv[0])+4*sizeof(double));

  valarray<TTest> va2(10);
  print(sizeof(TTest));
  print(sizeof(va2[0]));
  print(&va2[0]);
  print(&va2[9]);
  print(&va2[0]+9);
  print(long(&va2[9]));
  print(long(&va2[0])+9*sizeof(TTest));

  return 0;
}
//-------------------------------------------------------------------------------------------
