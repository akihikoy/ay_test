//-------------------------------------------------------------------------------------------
/*! \file    set_itr.cpp
    \brief   std::set of iterator.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <iomanip>
#include <algorithm>
#include <set>
#include <list>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*i;}
template<typename T> inline void PrintContainerR(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*(*i);}
#define printv(var) do{std::cout<<#var"= ";PrintContainer(var); std::cout<<std::endl;}while(0)
#define printvr(var) do{std::cout<<#var"= ";PrintContainerR(var); std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int src1[] = {2,1,3};

  std::list<int> list1(src1,src1+sizeof(src1)/sizeof(src1[0]));
  std::set<std::list<int>::iterator> set1;
  std::list<int>::iterator itr= list1.begin();
//   set1.insert(itr);
  ++itr;
//   set1.insert(itr);
// WARNING: We cannot use insert for a set of insert since Compare=less<T> is not defined for iterators.

  printv(list1);
  printvr(set1);
  print("------");

  return 0;
}
//-------------------------------------------------------------------------------------------
