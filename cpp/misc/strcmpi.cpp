//-------------------------------------------------------------------------------------------
/*! \file    strcmpi.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.29, 2011
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cstring>
int strcmpi(const char *s1, const char *s2)
{
  for(;*s1!='\0' && *s2!='\0';++s1,++s2)
  {
    char c1=*s1,c2=*s2;
    if(c1>='A'&&c1<='Z') c1-=('A'-'a');
    if(c2>='A'&&c2<='Z') c2-=('A'-'a');
    if(c1<c2)  return -1;
    if(c1>c2)  return +1;
  }
  if(*s1<*s2)  return -1;
  if(*s1>*s2)  return +1;
  return 0;
}
using namespace std;
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  if(argc!=3) {cerr<<"./a.out STR1 STR2"<<endl; return -1;}
  cout<<"strcmp : "<<argv[1]<<" "<<strcmp(argv[1],argv[2])<<" "<<argv[2]<<endl;
  cout<<"strcmpi: "<<argv[1]<<" "<<strcmpi(argv[1],argv[2])<<" "<<argv[2]<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
