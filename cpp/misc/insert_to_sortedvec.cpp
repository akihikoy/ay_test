//-------------------------------------------------------------------------------------------
/*! \file    insert_to_sortedvec.cpp
    \brief   Insert to sorted vector.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.27, 2018

g++ -g -Wall insert_to_sortedvec.cpp && ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*i;}
#define printv(var) do{std::cout<<#var"= ";PrintContainer(var); std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

/* Find an index idx of a sorted vector vec such that vec[idx]<=val<vec[idx+1].
  We can insert val by vec.insert(vec.begin()+idx+1, val) with keeping the sort.
  If vec.size()==0, idx=-1.
  If val<vec[0], idx=-1.
  If vec[vec.size()-1]<=val, idx=vec.size()-1. */
template<typename t_vector>
int FindIndex(const t_vector &vec, typename t_vector::const_reference val)
{
  for(int idx(0),end(vec.size()); idx<end; ++idx)
    if(val<vec[idx])  return idx-1;
  return vec.size()-1;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int src[] = {2,10,5,1,55,7,48,103,22,6,3,99,45,99};
  std::vector<int> vec1(src,src+sizeof(src)/sizeof(src[0]));
  printv(vec1);

  std::sort(vec1.begin(),vec1.end());
  printv(vec1);

  int idx;
  idx= FindIndex(vec1, 250);
  vec1.insert(vec1.begin()+idx+1, 250);
  idx= FindIndex(vec1, 88);
  vec1.insert(vec1.begin()+idx+1, 88);
  idx= FindIndex(vec1, 0);
  vec1.insert(vec1.begin()+idx+1, 0);
  printv(vec1);

  std::vector<int> vec2;
  printv(vec2);
  idx= FindIndex(vec2, 100);
  vec2.insert(vec2.begin()+idx+1, 100);
  printv(vec2);
  idx= FindIndex(vec2, 10);
  vec2.insert(vec2.begin()+idx+1, 10);
  printv(vec2);
  idx= FindIndex(vec2, 200);
  vec2.insert(vec2.begin()+idx+1, 200);
  printv(vec2);


  return 0;
}
//-------------------------------------------------------------------------------------------
