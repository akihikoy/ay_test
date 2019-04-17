#include <cmath>
// matlab-like mod function that returns always positive

// fast
template<typename T>
inline T Mod(const T &x, const T &y)
{
  if(y==0)  return x;
  return x-y*std::floor(x/y);
}

// very slow
// template<typename T>
// inline T Mod(const T &x, const T &y)
// {
//   T z= std::fmod(x,y);
//   if(x>=0)  return z;
//   else /*x<0*/ return (z==0) ? z : z+y;
// }

// (normal fmod) very slow
// template<typename T>
// inline T Mod(const T &x, const T &y)
// {
//   return std::fmod(x,y);
// }

// #include<iostream>
// using namespace std;
// int main()
// {
//   double y;
//   for(double x(-10);x<10;x+=0.1)
//   {
//     y= Mod(x,1.0);
//     cout<<y<<endl;
//   }
//   return 0;
// }

int main()
{
  double y;
  for(double x(-10000);x<10000;x+=0.00001)
    y= Mod(x,1.0);
  return 0;
}
