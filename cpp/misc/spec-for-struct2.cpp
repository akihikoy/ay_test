//-------------------------------------------------------------------------------------------
/*! \file    spec-for-struct.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.11, 2010
*/
//-------------------------------------------------------------------------------------------

#include <iostream>
#include <list>
using namespace std;

template <typename T>
struct TPrintObj
{
  void operator()(const T &x)
    {
      cout<<"value is "<<x<<endl;
    }
};

template <typename T>
void Print(const T &x)
{
  TPrintObj<T>()(x);
}

// template <typename T>
// void Print<list<T> >(const list<T> &x)
// {
  // cout<<"values(list) are ";
  // for(typename list<T>::const_iterator itr(x.begin()),last(x.end()); itr!=last; ++itr)
    // cout<< " " << *itr;
  // cout<<endl;
// }

template <>
struct TPrintObj<int>
{
  void operator()(const int &x)
  {
    cout<<"value(int) is "<<x<<endl;
  }
};

template <typename T>
struct TPrintObj<list<T> >
{
  void operator()(const list<T> &x)
  {
    cout<<"values(list) are ";
    for(typename list<T>::const_iterator itr(x.begin()),last(x.end()); itr!=last; ++itr)
      cout<< " " << *itr;
    cout<<endl;
  }
};

template <typename t_type>
struct TStruct
{
  typedef t_type T;
  T &Entity;
  TStruct(T &entity) : Entity(entity) {}
};

template <typename T>
TStruct<T> Struct(T &entity) {return TStruct<T>(entity);}

template <typename T>
struct TPrintObj<TStruct<T> >
{
  void operator()(const TStruct<T> &x)
    {
      cout<<"values are "<<endl;
      cout<<"  .X= "<<x.Entity.X<<endl;
      cout<<"  .Y= "<<x.Entity.Y<<endl;
    }
};


struct TItem
{
  int X;
  double Y;
};

int main(int argc, char**argv)
{
  list<double> tmp1; tmp1.push_back(1.2); tmp1.push_back(-5.5); tmp1.push_back(0.001);
  TItem tmp2= {-12, 4.55};
  Print(2.23);
  Print(-10);
  Print(tmp1);
  Print(Struct(tmp2));
  return 0;
}
//-------------------------------------------------------------------------------------------
