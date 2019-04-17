//-------------------------------------------------------------------------------------------
/*! \file    if-int.cpp
    \brief   Check if a string is int or not.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.19, 2017
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>  // strtol,atol
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

inline bool IsInt1(const std::string &s)
{
  if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))  return false;
  char *p ;
  strtol(s.c_str(), &p, 10);
  return (*p == 0) ;
}
inline bool IsInt2(const std::string &s)
{
  if(s.empty() || std::isspace(s[0]))  return false;
  char *p;
  strtol(s.c_str(), &p, 10);
  return (*p == 0);
}

int main(int argc, char**argv)
{
  if(argc==1)  
  {
    std::cerr<<"Input a string or a number as an argument."<<std::endl;
    return 0;
  }
  std::string str= argv[1];
  print(IsInt1(str));
  print(IsInt2(str));
  print(atol(str.c_str()));
  return 0;
}
//-------------------------------------------------------------------------------------------
