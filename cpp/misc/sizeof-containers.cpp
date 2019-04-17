#include <iostream>
#include <map>
#include <vector>
#include <list>
#include <string>
#include <boost/function.hpp>
using namespace std;

#define print(var) cout<<#var"= "<<(var)<<endl

int main(int,char**)
{
  print(sizeof(int));
  print(sizeof(double));
  print(sizeof(string));
  print(sizeof(map<int,int>));
  print(sizeof(map<double,int>));
  print(sizeof(map<string,int>));
  print(sizeof(vector<int>));
  print(sizeof(vector<double>));
  print(sizeof(vector<string>));
  print(sizeof(list<int>));
  print(sizeof(list<string>));
  print(sizeof(boost::function<void(void)>));
  print(sizeof(boost::function<void(int)>));
}
