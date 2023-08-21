//-------------------------------------------------------------------------------------------
/*! \file    str_split.cpp
    \brief   Split a string with a delimiter;
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.21, 2023

g++ -g -Wall -O2 str_split.cpp && ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
//-------------------------------------------------------------------------------------------
template<typename t_vec>
void PrintContainer(const t_vec &v)
{
  for(typename t_vec::const_iterator itr(v.begin()),itr_end(v.end()); itr!=itr_end; ++itr)
    std::cout<<" ### "<<*itr;
}
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define printv(var) std::cout<<#var"= ";PrintContainer(var);std::cout<<std::endl
//-------------------------------------------------------------------------------------------

// Split a string str by  delimiter delim.
void SplitString(const std::string &str, std::vector<std::string> &result, char delim=',')
{
  std::stringstream ss(str);
  result.clear();
  while(ss.good())
  {
    std::string substr;
    std::getline(ss, substr, delim);
    result.push_back(substr);
  }
}
// Split a string str by  delimiter delim.
std::vector<std::string> SplitString(const std::string &str, char delim=',')
{
  std::vector<std::string> result;
  SplitString(str, result, delim);
  return result;
}
//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  {
    std::vector<std::string> res1;
    SplitString("/tmp/file.txt,hoge.txt", res1);
    printv(res1);
  }
  {
    printv(SplitString("/tmp/file.txt,hoge.txt"));
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
