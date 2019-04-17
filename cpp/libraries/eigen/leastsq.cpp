//-------------------------------------------------------------------------------------------
/*! \file    leastsq.cpp
    \brief   Find W of W*x=y for {(x,y)}
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.24, 2015

    Usage:
      x++ -eig -slora leastsq.cpp
*/
//-------------------------------------------------------------------------------------------
#include <Eigen/Dense>
#include <cassert>
//-------------------------------------------------------------------------------------------
namespace trick
{

/* Find a least square solution of W in W*x=y for {(x,y)}.
    x= (x_1,...,x_d); d: dim of x.
    y= (y_1,...,y_c); c: dim of y.
    X= [x_1_1,...,x_d_1]
       [x_1_2,...,x_d_2]
       ...
       [x_1_N,...,x_d_N]; N: num of samples.
    Y= [y_1_1,...,y_c_1]
       [y_1_2,...,y_c_2]
       ...
       [y_1_N,...,y_c_N]; N: num of samples.
    lambda: regularization parameter.
    return W.
*/
Eigen::MatrixXd LeastSq(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const double &lambda=0.01)
{
  assert(X.rows()==Y.rows());
  Eigen::MatrixXd I= Eigen::MatrixXd::Identity(X.cols(),X.cols());
  return /*W=*/ ( (X.transpose()*X + lambda * I).inverse()*X.transpose() * Y ).transpose();
}
//-------------------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <lora/rand.h>
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<std::endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  Srand();
  {
    Eigen::MatrixXd X(100,2),Y(100,2);
    for(int i(0);i<100;++i)
    {
      X(i,0)= Rand(-1.0,1.0);
      X(i,1)= Rand(-1.0,1.0);
      Y(i,0)= 2.0*X(i,0) - 3.1*X(i,1) + 0.1*Rand(-1.0,1.0);
      Y(i,1)= 1.25*X(i,1)             + 0.1*Rand(-1.0,1.0);
    }
    Eigen::MatrixXd W= LeastSq(X,Y);
    print(W);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
