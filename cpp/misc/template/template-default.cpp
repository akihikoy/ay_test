//-------------------------------------------------------------------------------------------
/*! \file    template-default.cpp
    \brief   Template default parameter test
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.20, 2015
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

template <typename t_value=double, bool t_using_x=true>
struct TTest
{
  t_value X, Y;
  TTest() : X(1), Y(2) {}
  void Print()
    {
      if(t_using_x)
        std::cout<<"TYPE 1: "<<X<<std::endl;
      else
        std::cout<<"TYPE 2: "<<Y<<std::endl;
    }
};

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  // TTest test1;  // NG
  // TTest<double> test1;  // OK
  TTest<> test1;  // OK
  TTest<double,false> test2;
  test1.Print();
  test2.Print();
  return 0;
}
//-------------------------------------------------------------------------------------------
