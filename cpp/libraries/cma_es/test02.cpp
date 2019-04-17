//-------------------------------------------------------------------------------------------
/*! \file    test02.cpp
    \brief   Test of CMA-ES written in C - 2.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.22, 2016

cf. ../python/cma_es/testa1.py
*/
//-------------------------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib> /* free() */
#include "cma_es/cmaes_interface.h"
#include "cma_es/boundary_transformation.h"
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
  cmaes_t evo; /* an CMA-ES type struct or "object" */

  int Dim(2);
  /* initialize boundaries, be sure that initialSigma is smaller than upper minus lower bound */
  double bounds[2][2]= {{-3.0,-3.0}, {3.0,3.0}};
  cmaes_boundary_transformation_t boundaries;
  cmaes_boundary_transformation_init(&boundaries, bounds[0], bounds[1], /*len=*/Dim);
  /* Initialize everything into the struct evo, 0 means default */
  // Manual setup:
  double x0[]= {1.2,0.0};
  double sig0[]= {0.5,0.5};
  double *arFunvals= cmaes_init(&evo, /*dim=*/Dim, /*x0=*/x0, /*sig0=*/sig0, /*seed=*/0, /*lambda=*/0, /*param_file=*/"");
  // Manual parameter setup:
  evo.sp.stopMaxFunEvals= 22500;
  evo.sp.stopTolFun= 1e-8;  // 1e-12
  evo.sp.stopTolFunHist= 1e-9;  // 1e-13

  printf("%s\n", cmaes_SayHello(&evo));
  cmaes_ReadSignals(&evo, "cmaes_signals.par");  /* write header and initial values */

  // temporary value
  double *x_in_bounds= cmaes_NewDouble(Dim); /* calloc another vector */

  /* Iterate until stop criterion holds */
  while(!cmaes_TestForTermination(&evo))
  {
    /* generate lambda new search points, sample population */
    double *const *pop= cmaes_SamplePopulation(&evo); /* do not change content of pop */

    /* evaluate the new search points using fitfun */
    for(int i(0),i_end(cmaes_Get(&evo,"lambda")); i<i_end; ++i)
    {
      while(true)
      {
        cmaes_boundary_transformation(&boundaries, pop[i], x_in_bounds, Dim);
        double f; bool is_feasible;
        f= fobj1(x_in_bounds, is_feasible);
        if(is_feasible) {arFunvals[i]= f; break;}
        cmaes_ReSampleSingle(&evo, i);
      }
    }

    /* update the search distribution used for cmaes_SamplePopulation() */
    cmaes_UpdateDistribution(&evo, arFunvals);

    /* read instructions for printing output or changing termination conditions */
    cmaes_ReadSignals(&evo, "cmaes_signals.par");
  }
  free(x_in_bounds);
  printf("Stop:\n%s\n",  cmaes_TestForTermination(&evo)); /* print termination reason */
  cmaes_WriteToFile(&evo, "all", "/tmp/allcmaes.dat");         /* write final results */
  cmaes_WriteToFile(&evo, "all", "/dev/stdout");

  /* get best estimator for the optimum, xmean or xbestever */
  double *xfinal= cmaes_NewDouble(Dim);
  cmaes_boundary_transformation(&boundaries, (double const*)cmaes_GetPtr(&evo, "xbestever"), xfinal, Dim);
  cmaes_exit(&evo); /* release memory */
  cmaes_boundary_transformation_exit(&boundaries); /* release memory */

  /* do something with final solution and finally release memory */
  free(xfinal);

  return 0;
}
//-------------------------------------------------------------------------------------------
