//-------------------------------------------------------------------------------------------
/*! \file    copy-inherit.cpp.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jun.11, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
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

class TTest1
{
public:
  int A;
  double B;
  TTest1(int a, const double &b) : A(a), B(b) {}
  virtual void Print() {print(A); print(B);}
};
class TTest2 : public TTest1
{
public:
  double C;
  TTest2(int a, const double &b, const double &c) : TTest1(a,b),C(c) {}
  virtual void Print() {TTest1::Print(); print(C);}
};


int main(int argc, char**argv)
{
  TTest1 test1(2,3.2);
  TTest2 test2(1,-2.5,9.99);
  test1.Print();
  test2.Print();
  test1= test2;
  test1.Print();
  test2.Print();
  return 0;
}
//-------------------------------------------------------------------------------------------
