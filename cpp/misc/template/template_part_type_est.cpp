//-------------------------------------------------------------------------------------------
/*! \file    template_part_type_est.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.11, 2015
*/
//-------------------------------------------------------------------------------------------
using namespace std;
#include <iostream>
//-------------------------------------------------------------------------------------------
template <typename t_1, typename t_2>
inline t_1 Func(const t_2 &x)
{
  return t_1(x);
}
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  print((Func<int,double>(3.14)));
  print(Func<int>(3.14));  // WORKS!
  print(Func<double>(3.14));  // WORKS!
  return 0;
}
//-------------------------------------------------------------------------------------------
