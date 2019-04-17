#include <vector>
#include <string>
#include <iostream>
std::string IntArrayToStr(const std::vector<long> &a, int str_len)
{
  std::string s;
  s.resize(str_len);
  std::string::iterator sitr(s.begin());
  for(std::vector<long>::const_iterator aitr(a.begin()),aend(a.end()); aitr!=aend; ++aitr)
  {
    *sitr=((unsigned long)(*aitr)&0xFF000000)>>24; ++sitr; if(sitr==s.end()) break;
    *sitr=((unsigned long)(*aitr)&0x00FF0000)>>16; ++sitr; if(sitr==s.end()) break;
    *sitr=((unsigned long)(*aitr)&0x0000FF00)>>8;  ++sitr; if(sitr==s.end()) break;
    *sitr=((unsigned long)(*aitr)&0x000000FF);     ++sitr; if(sitr==s.end()) break;
  }
  return s;
}
void StrToIntArray(const std::string &s, std::vector<long> &a)
{
  a.resize((s.length()%4==0)?(s.length()/4):(s.length()/4+1));
  if(a.size()==0)  return;
  std::string::const_iterator sitr(s.begin());
  unsigned long c4(0);
  for(std::vector<long>::iterator aitr(a.begin()),aend(a.end()); aitr!=aend; ++aitr)
  {
    std::cout<<" "<<(unsigned long)(*sitr); c4 =((unsigned long)(*sitr))<<24; ++sitr; if(sitr==s.end()) break;
    std::cout<<" "<<(unsigned long)(*sitr); c4|=((unsigned long)(*sitr))<<16; ++sitr; if(sitr==s.end()) break;
    std::cout<<" "<<(unsigned long)(*sitr); c4|=((unsigned long)(*sitr))<<8;  ++sitr; if(sitr==s.end()) break;
    std::cout<<" "<<(unsigned long)(*sitr); c4|=((unsigned long)(*sitr));     ++sitr; if(sitr==s.end()) break;
    *aitr= c4;
  }
  a.back()=c4;
std::cout<<std::endl;
}

using namespace std;
int main(int argc,char**argv)
{
cout<<sizeof(char)<<endl;
cout<<sizeof(int)<<endl;
cout<<sizeof(long)<<endl;
  string s=argv[(argc>1)?1:0];
  vector<long> a;
  StrToIntArray(s,a);
  for(vector<long>::const_iterator i(a.begin()),e(a.end());i!=e;++i) cout<<" "<<*i;
  cout<<endl;
  cout<<"decoded: "<<IntArrayToStr(a, s.length())<<endl;
  return 0;
}
