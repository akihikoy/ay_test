//-------------------------------------------------------------------------------------------
/*! \file    bitfield.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.21, 2013
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void PrintBitField(unsigned char dat[], int size, int n=2)
{
  for(int d(0);d<size;++d)
  {
    unsigned char curr= dat[d];
    for(int j(0); j<8/n; ++j)
    {
      unsigned char res(0);
      res= (curr&0x80)/0x80;
      for(int i(1);i<n;++i)  res= res*2+((curr<<i)&0x80)/0x80;
      curr= curr<<n;
      cout<<std::hex<<(int)res;
    }
    cout<<" ";
  }
  cout<<endl;
}

void PrintBitField(unsigned short dat[], int size, int n=2)
{
  for(int d(0);d<size;++d)
  {
    unsigned short curr= dat[d];
    for(int j(0); j<16/n; ++j)
    {
      unsigned short res(0);
      res= (curr&0x8000)/0x8000;
      for(int i(1);i<n;++i)  res= res*2+((curr<<i)&0x8000)/0x8000;
      curr= curr<<n;
      cout<<std::hex<<(int)res;
    }
    cout<<" ";
  }
  cout<<endl;
}

void PrintHex(unsigned char dat[], int size)
{
  const unsigned char *b (dat);
  for(int i(size); i>0; --i,++b)
    cout<<" "<<std::hex<<std::setfill('0')<<std::setw(2)<<(int)(*b)<<std::dec;
  cout<<endl;
}


int main(int argc, char**argv)
{
  unsigned char dat[10];
  unsigned short val[8];
  print(sizeof(unsigned char));
  print(sizeof(unsigned short));
  for(int i(0);i<10;++i)  dat[i]= rand()%256;
  // memcpy(val,dat,10);

  PrintHex(dat,10);
  PrintBitField(dat,10,8);

  PrintBitField(dat,10,2);

  // 8 9 a b c
  // 1100 0000: 0xc0
  // 1111 0000: 0xf0
  // 0011 1111: 0x3f

  val[0]= ((dat[0]<<2))      + ((dat[1]&0xc0)>>6); // 8,2
  val[1]= ((dat[1]&0x3f)<<4) + ((dat[2]&0xf0)>>4); // 6,4
  val[2]= ((dat[2]&0x0f)<<6) + ((dat[3]&0xfc)>>2); // 4,6
  val[3]= ((dat[3]&0x03)<<8) + ((dat[4])); // 2,8

  val[4]= ((dat[5]<<2))      + ((dat[6]&0xc0)>>6); // 8,2
  val[5]= ((dat[6]&0x3f)<<4) + ((dat[7]&0xf0)>>4); // 6,4
  val[6]= ((dat[7]&0x0f)<<6) + ((dat[8]&0xfc)>>2); // 4,6
  val[7]= ((dat[8]&0x03)<<8) + ((dat[9])); // 2,8

  PrintBitField(val,8,2);

  return 0;
}
//-------------------------------------------------------------------------------------------
