#include <iostream>
#include <string>
using namespace std;

enum TEnum {e1=1,e2,e3};

struct TTest
{
  template <TEnum t_e>  TTest(int);  // NOTE: cannot use this constructor!!!
  // template<> TTest<e1>(int x)  {cout<<"x is "<<x<<endl;}
  TTest(int x)  {cout<<"x(int): "<<x<<endl;}
  TTest(string x)  {cout<<"x(str): "<<x<<endl;}
};

int main()
{
  TTest t1(10);
  TTest t2(5);
  TTest t3("hoge");
  return 0;
}
