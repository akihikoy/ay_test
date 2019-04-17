//-------------------------------------------------------------------------------------------
/*! \file    insert_to_sorted.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.07, 2014
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
// #include <string>
#include <algorithm>
#include <vector>
#include <list>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

// Insert value to container sorted in ascending order (<<<)
template <typename t_container>
inline void InsertAscending(t_container &container, const typename t_container::value_type &value)
{
  container.insert(std::find_if(container.begin(),container.end(),
      std::bind1st(std::less<typename t_container::value_type>(),value)),value);
}
//-------------------------------------------------------------------------------------------

// Insert value to container sorted in descending order (>>>)
template <typename t_container>
inline void InsertDescending(t_container &container, const typename t_container::value_type &value)
{
  container.insert(std::find_if(container.begin(),container.end(),
      std::bind2nd(std::less<typename t_container::value_type>(),value)),value);
}
//-------------------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
template<typename T> inline void PrintContainer(const T &x) {for(typename T::const_iterator i(x.begin()),e(x.end());i!=e;++i)std::cout<<" "<<*i;}
#define printv(var) do{std::cout<<#var"= ";PrintContainer(var); std::cout<<std::endl;}while(0)
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  int src[] = {2,10,5,1,55,7,48,103,22,6,3,99,45,99};
  std::vector<int> array(src,src+sizeof(src)/sizeof(src[0]));
  printv(array);

  std::sort(array.begin(),array.end());
  InsertAscending(array,20);
  InsertAscending(array,120);
  InsertAscending(array,0);
  printv(array);

  std::sort(array.rbegin(),array.rend());
  InsertDescending(array,-2);
  InsertDescending(array,3);
  InsertDescending(array,155);
  printv(array);

  std::list<int> array2,array3;
  while(true)
  {
    int a;
    cout<<"Type num (0 to exit) > ";
    cin>>a;
    if(a==0)  break;
    InsertAscending(array2,a);
    InsertDescending(array3,a);
  }
  printv(array2);
  printv(array3);

  return 0;
}
//-------------------------------------------------------------------------------------------
