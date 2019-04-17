#include <cmath>
#include <iostream>
using namespace std;
int main()
{
  for(double x(-5.0); x<5.0; x+=0.01)
    cout<<x<<" "<<round(x)<<endl;
  return 0;
}
