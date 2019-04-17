//-------------------------------------------------------------------------------------------
/*! \file    set_ops.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.27, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <iomanip>
#include <algorithm>
#include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
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


  std::vector<int> set1(src1,src1+sizeof(src1)/sizeof(src1[0]));
  std::vector<int> set2(src2,src2+sizeof(src2)/sizeof(src2[0]));
  std::vector<int> set_out;
  std::vector<int>::iterator last;

  printv(set1);
  printv(set2);

  print("------");

  std::sort(set1.begin(),set1.end());
  std::sort(set2.begin(),set2.end());

  printv(set1);
  printv(set2);

  print("union: set1 | set2");
  set_out.resize(set1.size()+set2.size());
  last= set_union(set1.begin(),set1.end(), set2.begin(),set2.end(), set_out.begin());
  set_out.resize(last-set_out.begin());
  printv(set_out);

  print("intersection: set1 & set2");
  set_out.resize(set1.size()+set2.size());
  last= set_intersection(set1.begin(),set1.end(), set2.begin(),set2.end(), set_out.begin());
  set_out.resize(last-set_out.begin());
  printv(set_out);

  print("difference: set1 - set2");
  set_out.resize(set1.size()+set2.size());
  last= set_difference(set1.begin(),set1.end(), set2.begin(),set2.end(), set_out.begin());
  set_out.resize(last-set_out.begin());
  printv(set_out);

  return 0;
}
//-------------------------------------------------------------------------------------------
