//-------------------------------------------------------------------------------------------
/*! \file    valarray4.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.18, 2013
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <iostream>
// #include <iomanip>
// #include <string>
#include <valarray>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

/*!\brief type extension class */
template <typename t_type>
struct TypeExt
{
  typedef typename t_type::value_type         value_type;
  typedef typename t_type::iterator           iterator;
  typedef typename t_type::const_iterator     const_iterator;
  typedef typename t_type::reference          reference;
  typedef typename t_type::const_reference    const_reference;
};

/*!\brief type extension class partially specialized for const t_type */
template <typename t_type>
struct TypeExt <const t_type>
{
  typedef typename TypeExt<t_type>::value_type         value_type;
  typedef typename TypeExt<t_type>::iterator           iterator;
  typedef typename TypeExt<t_type>::const_iterator     const_iterator;
  typedef typename TypeExt<t_type>::reference          reference;
  typedef typename TypeExt<t_type>::const_reference    const_reference;
};

/*!\brief type extension class partially specialized for valarray t_type */
template <typename t_type>
struct TypeExt <std::valarray<t_type> >
{
  typedef t_type         value_type;
  typedef t_type*        iterator;
  typedef const t_type*  const_iterator;
  typedef t_type&        reference;
  typedef const t_type&  const_reference;
};

/* generic functions */

template <typename t_type> inline typename TypeExt<t_type>::iterator GenBegin (t_type &x)              {return x.begin();}
template <typename t_type> inline typename TypeExt<t_type>::const_iterator GenBegin (const t_type &x)  {return x.begin();}
template <typename t_type> inline typename TypeExt<t_type>::iterator GenEnd (t_type &x)                {return x.end();}
template <typename t_type> inline typename TypeExt<t_type>::const_iterator GenEnd (const t_type &x)    {return x.end();}
template <typename t_type> inline typename TypeExt<t_type>::reference GenAt (t_type &x,int i)              {return x[i];}
template <typename t_type> inline typename TypeExt<t_type>::const_reference GenAt (const t_type &x,int i)  {return x[i];}
template <typename t_type> inline int   GenSize (const t_type &x)  {return x.size();}
template <typename t_type> inline void  GenResize (t_type &x, int s)  {return x.resize(s);}
template <typename t_type> inline void  GenResize (t_type &x, int s, const typename TypeExt<t_type>::value_type &vfill)  {return x.resize(s,vfill);}

// specialization for valarray

template <typename t_type> inline typename TypeExt<std::valarray<t_type> >::iterator GenBegin (std::valarray<t_type>  &x)              {return &x[0];}
template <typename t_type> inline typename TypeExt<std::valarray<t_type> >::const_iterator GenBegin (const std::valarray<t_type>  &x)  {return &x[0];}
template <typename t_type> inline typename TypeExt<std::valarray<t_type> >::iterator GenEnd (std::valarray<t_type>  &x)                {return &x[0]+x.size();}
template <typename t_type> inline typename TypeExt<std::valarray<t_type> >::const_iterator GenEnd (const std::valarray<t_type>  &x)    {return &x[0]+x.size();}
template <typename t_type> inline void  GenResize (std::valarray<t_type>  &x, int s, const t_type &vfill)  {return x.resize(s,vfill);}

template<typename T>
void print(const std::valarray<T> &x)
{
  for(typename TypeExt<std::valarray<T> >::const_iterator itr(GenBegin(x)),last(GenEnd(x));itr!=last;++itr)
    std::cout<<" "<<*itr;
  std::cout<<std::endl;
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  double xa[]= {1.,2.,3.,4.,5.};
  valarray<double> xv(xa,5);
  xv/=10.0;
  print(xv);
  return 0;
}
//-------------------------------------------------------------------------------------------
