//-------------------------------------------------------------------------------------------
/*! \file    binary.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.26, 2010
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
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <string>
// #include <vector>
#include <list>
#include <cstring>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

void write_to_osb_itr(std::ostreambuf_iterator<char> &osb_itr, const void *data_first, size_t len)
{
  const char *data_itr(reinterpret_cast<const char*>(data_first));
  for(;len>0;--len,++osb_itr,++data_itr)
    *osb_itr= *data_itr;
}
void read_from_isb_itr(std::istreambuf_iterator<char> &isb_itr, void *data_first, size_t len)
{
  char *data_itr(reinterpret_cast<char*>(data_first));
  for(;len>0;--len,++isb_itr,++data_itr)
    *data_itr= *isb_itr;
}
void readsome_from_isb_itr(std::istreambuf_iterator<char> &isb_itr, void *data_first, size_t len)
{
  std::istreambuf_iterator<char> isb_last;
  char *data_itr(reinterpret_cast<char*>(data_first));
  for(;len>0 && isb_itr!=isb_last;--len,++isb_itr,++data_itr)
    *data_itr= *isb_itr;
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

void SaveAsBinary1(const std::string &filename, const std::list<TData> &data_list)
{
  int size;
  std::ofstream ofs(filename.c_str());
  ofs.write(FILE_TYPE_ID,strlen(FILE_TYPE_ID));
  for (std::list<TData>::const_iterator itr(data_list.begin()),last(data_list.end()); itr!=last; ++itr)
  {
    switch(itr->K)
    {
    case dkI :
      ofs.write("i",1);
      ofs.write(reinterpret_cast<const char*>(&itr->I),sizeof(itr->I));
      break;
    case dkF :
      ofs.write("f",1);
      ofs.write(reinterpret_cast<const char*>(&itr->F),sizeof(itr->F));
      break;
    case dkS :
      ofs.write("s",1);
      size= itr->S.size();
      ofs.write(reinterpret_cast<const char*>(&size),sizeof(size));
      ofs.write(itr->S.c_str(),size);
      break;
    default:  throw;
    }
  }
  ofs.write("x",1);
  ofs.close();
}

void SaveAsBinary2(const std::string &filename, const std::list<TData> &data_list)
{
  int size;
  std::ofstream ofs(filename.c_str());
  std::ostreambuf_iterator<char> ofs_itr(ofs);
  write_to_osb_itr(ofs_itr,FILE_TYPE_ID,strlen(FILE_TYPE_ID));
  for (std::list<TData>::const_iterator itr(data_list.begin()),last(data_list.end()); itr!=last; ++itr)
  {
    switch(itr->K)
    {
    case dkI :
      *ofs_itr= 'i';  ++ofs_itr;
      write_to_osb_itr(ofs_itr,&itr->I,sizeof(itr->I));
      break;
    case dkF :
      *ofs_itr= 'f';  ++ofs_itr;
      write_to_osb_itr(ofs_itr,&itr->F,sizeof(itr->F));
      break;
    case dkS :
      *ofs_itr= 's';  ++ofs_itr;
      size= itr->S.size();
      write_to_osb_itr(ofs_itr,&size,sizeof(size));
      write_to_osb_itr(ofs_itr,itr->S.c_str(),size);
      break;
    default:  throw;
    }
  }
  *ofs_itr= 'x';  ++ofs_itr;
  ofs.close();
}

void LoadFromBinary1(const std::string &filename, std::list<TData> &data_list)
{
  char              ch_buf, block_buf[strlen(FILE_TYPE_ID)];
  TPrimitiveReader  pr_buf;
  std::ifstream ifs(filename.c_str());
  data_list.clear();
  ifs.readsome(block_buf,strlen(FILE_TYPE_ID));
  if (strncmp(block_buf,FILE_TYPE_ID,strlen(FILE_TYPE_ID))!=0)
    {std::cerr<<"invalid file type: "<<filename<<std::endl;  throw;}
  while(!ifs.eof())
  {
    ifs.get(ch_buf);
    switch(ch_buf)
    {
    case 'i' :
      ifs.read(pr_buf.Mem,sizeof(int));
      data_list.push_back(Data(pr_buf.I));
      break;
    case 'f' :
      ifs.read(pr_buf.Mem,sizeof(double));
      data_list.push_back(Data(pr_buf.D));
      break;
    case 's' :
      ifs.read(pr_buf.Mem,sizeof(int));
      data_list.push_back(TData());
      data_list.back().K= dkS;
      {
        std::string &s(data_list.back().S);
        s.resize(pr_buf.I);
        for (std::string::iterator s_itr(s.begin()),s_last(s.end()); s_itr!=s_last; ++s_itr)
        {
          ifs.get(ch_buf);
          *s_itr= ch_buf;
        }
      }
      break;
    case 'x' : return;
    }
  }
}

void LoadFromBinary2(const std::string &filename, std::list<TData> &data_list)
{
  char              ch_buf, block_buf[strlen(FILE_TYPE_ID)];
  TPrimitiveReader  pr_buf;
  std::ifstream ifs(filename.c_str());
  std::istreambuf_iterator<char> ifs_itr(ifs),ifs_last;
  data_list.clear();
  readsome_from_isb_itr(ifs_itr,block_buf,strlen(FILE_TYPE_ID));
  if (strncmp(block_buf,FILE_TYPE_ID,strlen(FILE_TYPE_ID))!=0)
    {std::cerr<<"invalid file type: "<<filename<<std::endl;  throw;}
  while(ifs_itr!=ifs_last)
  {
    ch_buf= *ifs_itr;  ++ifs_itr;
    switch(ch_buf)
    {
    case 'i' :
      read_from_isb_itr(ifs_itr,pr_buf.Mem,sizeof(int));
      data_list.push_back(Data(pr_buf.I));
      break;
    case 'f' :
      read_from_isb_itr(ifs_itr,pr_buf.Mem,sizeof(double));
      data_list.push_back(Data(pr_buf.D));
      break;
    case 's' :
      read_from_isb_itr(ifs_itr,pr_buf.Mem,sizeof(int));
      data_list.push_back(TData());
      data_list.back().K= dkS;
      {
        std::string &s(data_list.back().S);
        s.resize(pr_buf.I);
        for (std::string::iterator s_itr(s.begin()),s_last(s.end()); s_itr!=s_last; ++s_itr,++ifs_itr)
          *s_itr= *ifs_itr;
      }
      break;
    case 'x' : return;
    }
  }
  ifs.close();
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
  if (argc==1)
  {
    list<TData> test;
    for (int i(0); i<500000; ++i)
    {
      test.push_back(Data(2.236/3.0));
      test.push_back(Data(-3000000));
      test.push_back(Data(-1.28/3.0));
      // test.push_back(Data(string("hoge hoge")));
      test.push_back(Data(255));
      // test.push_back(Data(string("gegege")));
      test.push_back(Data(4.22e+50/3.0));
      // test.push_back(Data(string("xxx")));
    }
    // SaveAsBinary1("test.dat",test);
    SaveAsBinary2("test.dat",test);
    // cerr<<"---------"<<endl;
    // PrintDataList(test);
    // cerr<<"---------"<<endl;
    cerr<<"saved to test.dat"<<endl;
    cerr<<"write "<<test.size()<<endl;
  }
  else
  {
    list<TData> test;
    // LoadFromBinary1(argv[1],test);
    LoadFromBinary2(argv[1],test);
    cerr<<"read "<<test.size()<<endl;
    cerr<<"loaded from "<<argv[1]<<endl;
    // cerr<<"---------"<<endl;
    // PrintDataList(test);
    // cerr<<"---------"<<endl;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
