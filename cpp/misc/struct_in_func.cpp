//-------------------------------------------------------------------------------------------
/*! \file    struct_in_func.cpp
    \brief   Test of defining a struct in a function.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.22, 2023

g++ -Wall -O2 struct_in_func.cpp -o struct_in_func.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
void Func1(int x)
{
  struct t_internal
  {
    void Print(int y)
      {
        std::cout<<"Func1:t_internal:Print:"<<y<<std::endl;
      }
  };
  t_internal intrnl;
  intrnl.Print(x);
}
#if 0
void Func2(int x)
{
  struct t_internal
  {
    void Print(int y)
      {
        // ERROR: use of parameter from containing function
        std::cout<<"Func2:t_internal:Print:"<<x+y<<std::endl;
      }
  };
  t_internal intrnl;
  intrnl.Print(200);
}
#endif
void Func3(int x)
{
  struct t_internal
  {
    int &x_;
    t_internal(int x) : x_(x) {}
    void Print(int y)
      {
        std::cout<<"Func3:t_internal:Print:"<<x_+y<<std::endl;
      }
  };
  t_internal intrnl(x);
  intrnl.Print(200);
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  Func1(100);
  // Func2(100);
  Func3(100);
  return 0;
}
//-------------------------------------------------------------------------------------------
