//-------------------------------------------------------------------------------------------
/*! \file    test03.cpp
    \brief   Good interface, equivalent to test02.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.22, 2016
*/
//-------------------------------------------------------------------------------------------
#include "cma_es/test03.h"
//-------------------------------------------------------------------------------------------
namespace trick
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

template <typename T>
inline T Sq(const T &x)  {return x*x;}

/* the objective (fitness) function to be minimized */
double fobj1(double const *x, bool &is_feasible)
{
  if(Sq(x[0]-0.5)+Sq(x[1]+0.5)<0.2)
  {
    is_feasible= false;
    return 0.0;
  }
  is_feasible= true;
  return 3.0*Sq(x[0]-1.2) + 2.0*Sq(x[1]+2.0);
}

int main(int argc, char**argv)
{
  TCMAESParams params;
  params.stopMaxFunEvals= 2000;
  params.stopTolFun= 1.0e-6;
  params.stopTolFunHist= 1.0e-7;
  // params.diagonalCov= 1.0;
  params.PrintLevel= 2;

  int Dim(2);
  double bounds[2][2]= {{-3.0,-3.0}, {3.0,3.0}};
  double x0[]= {1.2,0.0};
  double sig0[]= {0.5,0.5};
  double xres[2];
  MinimizeF(fobj1, x0, sig0, Dim, bounds[0], bounds[1], /*bound_len=*/Dim, xres, params);

  std::cout<<"\nxres=";
  for(int d(0);d<Dim;++d)  std::cout<<" "<<xres[d];
  std::cout<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
