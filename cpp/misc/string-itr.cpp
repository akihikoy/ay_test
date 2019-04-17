#include <string>
#include <iostream>
using namespace std;

int main(int,char**)
{
  string str="hoge hoge";
  // for (string::iterator itr(str.begin()); itr!=str.end(); ++itr)
    // *itr='+';
  for (string::const_iterator itr(str.begin()); itr!=str.end(); ++itr)
    if (*itr!=' ')
      cout<<" "<<*itr;
  cout<<endl;
  return 0;
}

