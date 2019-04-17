#include <cmath>

// cf. mod.cpp

// Matlab-like mod function that returns always positive
template<typename T>
inline T Mod(const T &x, const T &y)
{
  if(y==0)  return x;
  return x-y*std::floor(x/y);
}
// For integer type
// ref. http://stackoverflow.com/questions/14997165/fastest-way-to-get-a-positive-modulo-in-c-c
template<>
inline int Mod<int>(const int &x, const int &y)
{
  return (x%y+y)%y;
}

#include<iostream>
using namespace std;
int main()
{
 for(double x(-10);x<10;x+=0.1)
 {
   double y= Mod(x,3.0);
   cout<<x<<" "<<y<<endl;
 }
 for(int x(-10);x<10;++x)
 {
   int y= Mod(x,3);
   cout<<x<<" "<<y<<endl;
 }
 return 0;
}

