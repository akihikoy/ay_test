//-------------------------------------------------------------------------------------------
/*! \file    join_path.cpp
    \brief   Join a base dir and a vector of path.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.21, 2023

g++ -g -Wall -O2 join_path.cpp && ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
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

// Make a vector of path: [base_dir+f for f in file_names].
inline std::vector<std::string> PathJoin(const std::string base_dir, const std::vector<std::string> &file_names)
{
  std::string delim;
  std::vector<std::string> result;
  for(std::vector<std::string>::const_iterator fitr(file_names.begin()),fitr_end(file_names.end());
      fitr!=fitr_end; ++fitr)
  {
    if((base_dir.size()>0&&base_dir.back()=='/') || (fitr->size()>0&&fitr->front()=='/'))
      delim= "";
    else
      delim= "/";
    result.push_back(base_dir+delim+*fitr);
  }
  return result;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string base_dir1(".");
  std::string base_dir2("/home/ay");
  std::string base_dir3("/home/hoge/");
  std::string base_dir4("");
  std::vector<std::string> file_names;
  file_names.push_back("hoge.dat");
  file_names.push_back("/aa.dat");
  print(base_dir1);
  print(base_dir2);
  print(base_dir3);
  print(base_dir4);
  printv(file_names);

  printv(PathJoin(base_dir1, file_names));
  printv(PathJoin(base_dir2, file_names));
  printv(PathJoin(base_dir3, file_names));
  printv(PathJoin(base_dir4, file_names));
  return 0;
}
//-------------------------------------------------------------------------------------------
