//-------------------------------------------------------------------------------------------
/*! \file    list-erase2.cpp
    \brief   Erase list contents;
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <algorithm>
#include <list>
//-------------------------------------------------------------------------------------------
// namespace loco_rabbits
// {
// }
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
// using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*i;}
template<typename T> inline void PrintContainer(T i, T e) {for(;i!=e;++i)std::cout<<" "<<*i;}
#define printv(var) do{std::cout<<#var"= ";PrintContainer(var); std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define exec(exp) std::cout<<#exp<<": "; exp; std::cout<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::list<int> a;
  printv(a);
  print(a.begin()==a.end());
  print("=============");

  for(int i(1);i<=10;++i) {a.push_back(i);if(i==3||i==7)a.push_back(0);}
  printv(a);
  exec(std::list<int>::iterator itr((++std::find(a.rbegin(),a.rend(),0)).base()));
  exec(PrintContainer(itr,a.end()));
  exec(a.erase(itr,a.end()));
  printv(a);
  print("=============");

  print("removing 3,4,5");
  printv(a);
  // NOTE: This operation is the same as a.remove_if(lambda itr: *itr in [3,4,5]),
  // if C++ provides {lambda itr: *itr in [3,4,5]}.
  // Otherwise we need to define a function, which reduces the readability.
  // Rather than that, this will be better.
  for(std::list<int>::iterator itr(a.end()); itr!=a.begin();)
  {
    --itr;
    if(3<=*itr && *itr<=5)  itr= a.erase(itr);
  }
  // The same:
//   for(std::list<int>::iterator itr(a.begin()); itr!=a.end();)
//   {
//     if(3<=*itr && *itr<=5)  itr= a.erase(itr);
//     else ++itr;
//   }
  printv(a);
  return 0;
}
//-------------------------------------------------------------------------------------------
