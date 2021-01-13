//-------------------------------------------------------------------------------------------
/*! \file    map_of_pair.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <map>
//-------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream &os, const std::pair<int,int> &v)
{
  os<<"<"<<v.first<<","<<v.second<<">";
  return os;
}
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<i->first<<":"<<i->second;}
#define printv(var) do{std::cout<<#var"= {";PrintContainer(var); std::cout<<"}"<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::map<std::pair<int,int>, double> m1;
  printv(m1);
  m1[std::pair<int,int>(1,2)]= 1.2;
  m1[std::pair<int,int>(3,2)]= 3.2;
  m1[std::pair<int,int>(1,0)]= 1.0;
  printv(m1);
  print((m1[std::pair<int,int>(3,2)]));
  print((m1[std::pair<int,int>(2,2)]));
  printv(m1);
  print((m1.find(std::pair<int,int>(2,2))==m1.end()));
  print((m1.find(std::pair<int,int>(3,3))==m1.end()));
  return 0;
}
//-------------------------------------------------------------------------------------------
