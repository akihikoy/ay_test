//-------------------------------------------------------------------------------------------
/*! \file    binary-test.h
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.28, 2010
*/
//-------------------------------------------------------------------------------------------
#ifndef binary_test_h
#define binary_test_h
//-------------------------------------------------------------------------------------------
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <string>
#include <list>
#include <cstring>
#include "base64-test.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------


template <typename t_ostream_iterator>
inline void WriteToOutStreamIterator (t_ostream_iterator &os_itr, const void *data_first, size_t len)
{
  const char *data_itr(reinterpret_cast<const char*>(data_first));
  for(;len>0;--len,++os_itr,++data_itr)
    *os_itr= *data_itr;
}

template <typename t_istream_iterator>
inline void ReadFromInStreamIterator(t_istream_iterator &is_itr, void *data_first, size_t len)
{
  char *data_itr(reinterpret_cast<char*>(data_first));
  for(;len>0;--len,++is_itr,++data_itr)
    *data_itr= *is_itr;
}

template <typename t_istream_iterator>
inline void ReadsomeFromInStreamIterator(t_istream_iterator &is_itr, void *data_first, size_t len)
{
  t_istream_iterator is_last;
  char *data_itr(reinterpret_cast<char*>(data_first));
  for(;len>0 && is_itr!=is_last;--len,++is_itr,++data_itr)
    *data_itr= *is_itr;
}

enum TDataKind {dkI=0,dkF,dkS};

struct TData
{
  TDataKind    K;
  int          I;
  double       F;
  std::string  S;
};

TData Data(const int         &v)  {TData d; d.K= dkI; d.I= v;  return d;}
TData Data(const double      &v)  {TData d; d.K= dkF; d.F= v;  return d;}
TData Data(const std::string &v)  {TData d; d.K= dkS; d.S= v;  return d;}

union TPrimitiveReader
{
  char         C;
  int          I;
  float        F;
  double       D;
  long double  LD;
  char         Mem[128];
};

static const char * const FILE_TYPE_ID("TEST DAT");
static const int          FILE_TYPE_ID_LEN(8);

void PrintDataList(const std::list<TData> &data_list)
{
  for (std::list<TData>::const_iterator itr(data_list.begin()),last(data_list.end()); itr!=last; ++itr)
  {
    switch(itr->K)
    {
    case dkI :
      std::cout<<"I: "<<itr->I<<std::endl;
      break;
    case dkF :
      std::cout<<"F: "<<std::setprecision(15)<<itr->F<<std::endl;
      break;
    case dkS :
      std::cout<<"S: "<<itr->S<<std::endl;
      break;
    default:  throw;
    }
  }
}

template <typename t_ostream_iterator>
void SaveToStream(t_ostream_iterator &os_itr, const std::list<TData> &data_list)
{
  int size;
  WriteToOutStreamIterator(os_itr,FILE_TYPE_ID,FILE_TYPE_ID_LEN);
  for (std::list<TData>::const_iterator itr(data_list.begin()),last(data_list.end()); itr!=last; ++itr)
  {
    switch(itr->K)
    {
    case dkI :
      *os_itr= 'i';  ++os_itr;
      WriteToOutStreamIterator(os_itr,&itr->I,sizeof(itr->I));
      break;
    case dkF :
      *os_itr= 'f';  ++os_itr;
      WriteToOutStreamIterator(os_itr,&itr->F,sizeof(itr->F));
      break;
    case dkS :
      *os_itr= 's';  ++os_itr;
      size= itr->S.size();
      WriteToOutStreamIterator(os_itr,&size,sizeof(size));
      WriteToOutStreamIterator(os_itr,itr->S.c_str(),size);
      break;
    default:  throw;
    }
  }
  *os_itr= 'x';  ++os_itr;
}

void SaveAsBinary(const std::string &filename, const std::list<TData> &data_list)
{
  std::ofstream ofs(filename.c_str());
  std::ostreambuf_iterator<char> ofs_itr(ofs);

  SaveToStream(ofs_itr, data_list);

  ofs.close();
}

void SaveAsBase64(const std::string &filename, const std::list<TData> &data_list)
{
  std::ofstream ofs(filename.c_str());
  std::ostreambuf_iterator<char> ofs_itr(ofs);
  TOutBase64Iterator <std::ostreambuf_iterator<char> >  base64encoder(&ofs_itr);

  SaveToStream(base64encoder, data_list);

  ofs.close();
}

template <typename t_istream_iterator>
void LoadFromStream(t_istream_iterator &is_itr, std::list<TData> &data_list)
{
  char              ch_buf, block_buf[FILE_TYPE_ID_LEN];
  TPrimitiveReader  pr_buf;
  t_istream_iterator is_last;
  data_list.clear();
  ReadsomeFromInStreamIterator(is_itr,block_buf,FILE_TYPE_ID_LEN);
  if (strncmp(block_buf,FILE_TYPE_ID,FILE_TYPE_ID_LEN)!=0)
    {std::cerr<<"invalid stream type"<<std::endl;  throw;}
  while(is_itr!=is_last)
  {
    ch_buf= *is_itr;  ++is_itr;
    switch(ch_buf)
    {
    case 'i' :
      ReadFromInStreamIterator(is_itr,pr_buf.Mem,sizeof(int));
      data_list.push_back(Data(pr_buf.I));
      break;
    case 'f' :
      ReadFromInStreamIterator(is_itr,pr_buf.Mem,sizeof(double));
      data_list.push_back(Data(pr_buf.D));
      break;
    case 's' :
      ReadFromInStreamIterator(is_itr,pr_buf.Mem,sizeof(int));
      data_list.push_back(TData());
      data_list.back().K= dkS;
      {
        std::string &s(data_list.back().S);
        s.resize(pr_buf.I);
        for (std::string::iterator s_itr(s.begin()),s_last(s.end()); s_itr!=s_last; ++s_itr,++is_itr)
          *s_itr= *is_itr;
      }
      break;
    case 'x' : return;
    }
  }
}

void LoadFromBinary(const std::string &filename, std::list<TData> &data_list)
{
  std::ifstream ifs(filename.c_str());
  std::istreambuf_iterator<char> ifs_itr(ifs);

  LoadFromStream(ifs_itr, data_list);

  ifs.close();
}


void LoadFromBase64(const std::string &filename, std::list<TData> &data_list)
{
  std::ifstream ifs(filename.c_str());
  std::istreambuf_iterator<char> ifs_itr(ifs);
  TInBase64Iterator <std::istreambuf_iterator<char> >  base64decoder(&ifs_itr);

  LoadFromStream(base64decoder, data_list);

  ifs.close();
}



//-------------------------------------------------------------------------------------------
}  // end of namespace loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // binary_test_h
//-------------------------------------------------------------------------------------------
