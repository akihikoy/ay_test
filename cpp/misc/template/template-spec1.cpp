#include <iostream>
using namespace std;

enum TEnum {e1=1,e2,e3};

template <TEnum t_e>
void func()
{
  cout<<"template argument is: "<<static_cast<int>(t_e)<<endl;
}
template <>
void func<e2>()
{
  cout<<"specialized to e2"<<endl;
}

int main()
{
  func<e1>();
  func<e2>();
  func<e3>();
  return 0;
}
