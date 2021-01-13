//-------------------------------------------------------------------------------------------
/*! \file    set_std.cpp
    \brief   Test of std::set.  cf. set_ops.cpp
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <iomanip>
#include <algorithm>
#include <set>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// template<typename t_itr> inline void PrintContainer(t_itr i, t_itr e) {for(;i!=e;++i)std::cout<<" "<<*i;}
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*i;}
#define printv(var) do{std::cout<<#var"= ";PrintContainer(var); std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int src1[] = {2,1,3};
  int src2[] = {5,4,2};
  int src3[] = {5,10,15,20};

  std::set<int> set1(src1,src1+sizeof(src1)/sizeof(src1[0]));
  std::set<int> set2(src2,src2+sizeof(src2)/sizeof(src2[0]));

  printv(set1);
  printv(set2);

  print(set1.size());
  print(set1.size());
  // print(*(set1.begin()+3));  // does not work.
  std::set<int>::iterator itr= set1.begin();
  std::advance(itr,3);
  print(*itr);

  print("------");

  print("insert 2, 10, set2 into set1:");
  set1.insert(2);
  printv(set1);
  set1.insert(10);
  printv(set1);
  set1.insert(set2.begin(),set2.end());
  printv(set1);
  printv(set2);

  print("------");

  std::set<int> set3(src3,src3+sizeof(src3)/sizeof(src3[0]));

  print("intersection: set1 & set3");
  printv(set1);
  printv(set3);
  std::set<int> set_out;
  std::set_intersection(set1.begin(),set1.end(), set3.begin(),set3.end(), std::inserter(set_out,set_out.begin()));
  printv(set_out);

  print("difference: set1 - set2");
  printv(set1);
  printv(set2);
  set_out.clear();
  std::set_difference(set1.begin(),set1.end(), set2.begin(),set2.end(), std::inserter(set_out,set_out.begin()));
  printv(set_out);
  print("difference: set2 - set1");
  set_out.clear();
  std::set_difference(set2.begin(),set2.end(), set1.begin(),set1.end(), std::inserter(set_out,set_out.begin()));
  printv(set_out);

  return 0;
}
//-------------------------------------------------------------------------------------------
