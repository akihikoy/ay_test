#include <iostream>

struct TTest
{
  int X,Y;
};
inline TTest Test(int x,int y)
{
  TTest res;
  res.X=x; res.Y=y;
  return res;
}
std::ostream& operator<<(std::ostream &lhs, const TTest &rhs)
{
  lhs<<rhs.X<<", "<<rhs.Y;
  return lhs;
}

inline TTest operator_common(const TTest &lhs, const TTest &rhs, int(*op)(const int&,const int&))
{
  TTest res;
  res.X=op(lhs.X,rhs.X);
  res.Y=op(lhs.Y,rhs.Y);
  return res;
}
template<typename T> inline T operator_plus(const T&lhs,const T&rhs)  {return lhs+rhs;}
template<typename T> inline T operator_minus(const T&lhs,const T&rhs)  {return lhs-rhs;}
TTest operator+(const TTest &lhs, const TTest &rhs)
{
  return operator_common(lhs,rhs,operator_plus<int>);
}
TTest operator-(const TTest &lhs, const TTest &rhs)
{
  return operator_common(lhs,rhs,operator_minus<int>);
}

#define print(x) std::cout<<#x"= "<<(x)<<std::endl;
int main()
{
  TTest x=Test(2,3),y=Test(10,-3);
  print(x);
  print(y);
  print(x+y);
  print(x-y);
  return 0;
}

