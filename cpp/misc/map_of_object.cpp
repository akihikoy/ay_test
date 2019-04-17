//-------------------------------------------------------------------------------------------
/*! \file    map_of_object.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.17, 2015
*/
//-------------------------------------------------------------------------------------------
#include <map>
#include <string>
#include <iostream>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------

int ADynamicObject(1000);

// class TObject
// {
// public:
  // explicit TObject(const int idx=0)
      // : index_(idx), pointer_(&ADynamicObject)
    // {
      // cout<<"creating   \t"<<index_<<"\t"<<pointer_<<endl;
    // }
  // TObject(const TObject &rhs)
      // : index_(rhs.index_), pointer_(rhs.pointer_)
    // {
      // cout<<"copying    \t"<<index_<<"\t"<<pointer_<<endl;
    // }
  // virtual ~TObject()
    // {
      // cout<<"  deleting \t"<<index_<<"\t"<<pointer_<<endl;
    // }
  // virtual void Print()
    // {
      // cout<<"  this is  \t"<<index_<<"\t"<<pointer_<<endl;
    // }
// protected:
  // int index_;
  // int *pointer_;
// };

class TObject
{
public:
  explicit TObject()
      : index_(-1), pointer_(NULL)
    {
      cout<<"creating   \t"<<index_<<"\t"<<pointer_<<endl;
    }
  TObject(const TObject &rhs)
      : index_(rhs.index_), pointer_(rhs.pointer_)
    {
      cout<<"copying    \t"<<index_<<"\t"<<pointer_<<endl;
    }
  virtual ~TObject()
    {
      cout<<"  deleting \t"<<index_<<"\t"<<pointer_<<endl;
    }
  void Init(int idx, int *ptr)
    {
      index_= idx;
      pointer_= ptr;
      cout<<"  initting \t"<<index_<<"\t"<<pointer_<<endl;
    }
  virtual void Print()
    {
      cout<<"  this is  \t"<<index_<<"\t"<<pointer_<<endl;
    }
protected:
  int index_;
  int *pointer_;
};

//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define print(expr) std::cout<<#expr" : "; expr
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  map<string,TObject> data;
  // print(data["a"]= TObject(0));
  // print(data["b"]= TObject(1));
  // print(data.insert(std::pair<string,TObject>("a",TObject(0))));
  // print(data.insert(std::pair<string,TObject>("b",TObject(1))));
  print(data["a"]= TObject());
  print(data["a"].Init(0,&ADynamicObject));
  print(data["b"]= TObject());
  print(data["b"].Init(1,&ADynamicObject));
  print(data["a"].Print());
  print(data["b"].Print());
  for(map<string,TObject>::iterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
    {std::cout<<itr->first<<" : "; print(itr->second.Print());}
  for(map<string,TObject>::iterator itr(data.end()),itr_begin(data.begin()); itr!=itr_begin;)
    {--itr; std::cout<<itr->first<<" : "; print(itr->second.Print());}
  print(data.erase("a"));
  print(data.erase("b"));
  return 0;
}
//-------------------------------------------------------------------------------------------
