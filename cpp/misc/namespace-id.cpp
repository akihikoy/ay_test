#include <iostream>
namespace A
{
namespace B
{
  int AB(1);
}
}
namespace C
{
namespace B
{
  int CB(100);
}
}

using namespace std;
int main()
{
  cout<<A::B::AB<<endl;
  cout<<C::B::CB<<endl;
  return 0;
}
