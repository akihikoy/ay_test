//-------------------------------------------------------------------------------------------
/*! \file    function_ptr.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.04, 2020

g++ -Wall -ansi function_ptr.cpp
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------

void f1(void(*f)(void)=NULL)
{
  if(f)
  {
    std::cout<<"f1: f is given!"<<std::endl;
    f();
  }
  else
  {
    std::cout<<"f1: we don't have f."<<std::endl;
  }
}

void f2(void)
{
  std::cout<<"f2: this is f2."<<std::endl;
}

int main(int argc, char**argv)
{
  f1();
  std::cout<<"----------"<<std::endl;
  f1(f2);
  std::cout<<"----------"<<std::endl;
  {
    void(*f)(void)= NULL;
    if(f)
    {
      std::cout<<"main1: f is given!"<<std::endl;
      f();
    }
    else
    {
      std::cout<<"main1: we don't have f."<<std::endl;
    }
  }
  {
    void(*f)(void)= f2;
    if(f)
    {
      std::cout<<"main2: f is given!"<<std::endl;
      f();
    }
    else
    {
      std::cout<<"main2: we don't have f."<<std::endl;
    }
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
