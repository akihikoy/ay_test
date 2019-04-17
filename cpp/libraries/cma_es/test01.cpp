//-------------------------------------------------------------------------------------------
/*! \file    test01.cpp
    \brief   Test of CMA-ES written in C - 1.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.22, 2016

cf. ../python/cma_es/test6.py
*/
//-------------------------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib> /* free() */
#include "cma_es/cmaes_interface.h"
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
//-------------------------------------------------------------------------------------------

template <typename T>
inline T Sq(const T &x)  {return x*x;}

/* the objective (fitness) function to be minimized */
double fobj1(double const *x)
{
  return 3.0*(Sq(x[0]-1.5)+Sq(x[1]-0.5))+2.0;
}

int main(int argc, char**argv)
{
  cmaes_t evo; /* an CMA-ES type struct or "object" */

  /* Initialize everything into the struct evo, 0 means default */
  // double *arFunvals= cmaes_init(&evo, /*dim=*/0, /*x0=*/NULL, /*sig0=*/NULL, /*seed=*/0, /*lambda=*/0, "cmaes_initials.par");
  // // Manual setup:
  // int Dim(2);
  // double *x0= cmaes_NewDouble(/*dim=*/Dim);
  // x0[0]=0.1; x0[1]=0.1;
  // double *sig0= cmaes_NewDouble(/*dim=*/Dim);
  // sig0[0]=0.5; sig0[1]=0.5;
  // double *arFunvals= cmaes_init(&evo, /*dim=*/Dim, /*x0=*/x0, /*sig0=*/sig0, /*seed=*/0, /*lambda=*/0, /*param_file=*/"");
  // free(x0);
  // free(sig0);
  // Manual setup (easier):
  int Dim(2);
  double x0[]= {0.1,0.1};
  double sig0[]= {0.5,0.5};
  double *arFunvals= cmaes_init(&evo, /*dim=*/Dim, /*x0=*/x0, /*sig0=*/sig0, /*seed=*/0, /*lambda=*/0, /*param_file=*/"");
  // Manual parameter setup:
  evo.sp.stopMaxFunEvals= 22500;
  evo.sp.stopTolFun= 1e-8;  // 1e-12
  evo.sp.stopTolFunHist= 1e-9;  // 1e-13

  printf("%s\n", cmaes_SayHello(&evo));
  cmaes_ReadSignals(&evo, "cmaes_signals.par");  /* write header and initial values */

  /* Iterate until stop criterion holds */
  while(!cmaes_TestForTermination(&evo))
  {
    /* generate lambda new search points, sample population */
    double *const *pop= cmaes_SamplePopulation(&evo); /* do not change content of pop */

    /* Here we may resample each solution point pop[i] until it
                becomes feasible. function is_feasible(...) needs to be
                user-defined.
                Assumptions: the feasible domain is convex, the optimum is
                not on (or very close to) the domain boundary, initialX is
                feasible and initialStandardDeviations are sufficiently small
                to prevent quasi-infinite looping. */
    /* for (int i(0); i<cmaes_Get(&evo, "popsize"); ++i)
          while (!is_feasible(pop[i]))
            cmaes_ReSampleSingle(&evo, i);
    */

    /* evaluate the new search points using fitfun */
    for(int i(0),i_end(cmaes_Get(&evo,"lambda")); i<i_end; ++i)
    {
      arFunvals[i]= fobj1(pop[i]);
    }

    /* update the search distribution used for cmaes_SamplePopulation() */
    cmaes_UpdateDistribution(&evo, arFunvals);

    /* read instructions for printing output or changing termination conditions */
    cmaes_ReadSignals(&evo, "cmaes_signals.par");
  }
  printf("Stop:\n%s\n",  cmaes_TestForTermination(&evo)); /* print termination reason */
  cmaes_WriteToFile(&evo, "all", "/tmp/allcmaes.dat");         /* write final results */
  cmaes_WriteToFile(&evo, "all", "/dev/stdout");

  /* get best estimator for the optimum, xmean or xbestever */
  double *xfinal = cmaes_GetNew(&evo, "xbestever");
  cmaes_exit(&evo); /* release memory */

  /* do something with final solution and finally release memory */
  free(xfinal);

  return 0;
}
//-------------------------------------------------------------------------------------------
