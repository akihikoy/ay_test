//-------------------------------------------------------------------------------------------
/*! \file    base64-test.h
    \brief   certain c++ header file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.28, 2010
*/
//-------------------------------------------------------------------------------------------
#ifndef base64_test_h
#define base64_test_h
//-------------------------------------------------------------------------------------------
#include <cctype>


#include <cstdlib>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------


static const char INT_TO_BASE64[]=
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/=";  // note: final character [64] is used for padding

inline bool IsBase64(char c)
{
  return isalnum(c) || c==INT_TO_BASE64[62] || c==INT_TO_BASE64[63];
}

inline char Base64ToInt(char c)
{
  if('A'<=c && c<='Z')  return c-'A';
  if('a'<=c && c<='z')  return 26+(c-'a');
  if('0'<=c && c<='9')  return 52+(c-'0');
  if(c==INT_TO_BASE64[62])  return 62;
  if(c==INT_TO_BASE64[63])  return 63;
  if(c==INT_TO_BASE64[64])  return -1;
  // return -1;
std::exit(1);
}

template <typename t_ostream_iterator>
class TOutBase64Iterator
{
public:

  TOutBase64Iterator(t_ostream_iterator *os_itr) : out_stream_itr_(os_itr), idx_(0), written_(false) {}

  ~TOutBase64Iterator() {Flush();}

  char& operator*()  {written_=true; return buf_[idx_];}

  const TOutBase64Iterator& operator++()
    {
      ++idx_;
      written_= false;
      if(idx_==3)  Flush();
      return *this;
    }

  void Flush();

private:

  t_ostream_iterator *out_stream_itr_;

  char buf_[3];
  int  idx_;
  bool written_;

};

template <typename t_ostream_iterator>
void TOutBase64Iterator<t_ostream_iterator>::Flush()
{
  if(idx_==0 && !written_)  return;
  if(!written_)  --idx_;
  if(idx_==0)       {buf_[1]=0; buf_[2]=0;}
  else if(idx_==1)  {buf_[2]=0;}

  char encoded[4];
  encoded[0]= INT_TO_BASE64[(buf_[0] & 0xfc) >> 2];
  encoded[1]= INT_TO_BASE64[((buf_[0] & 0x03) << 4) + ((buf_[1] & 0xf0) >> 4)];
  if(idx_>=1) encoded[2]= INT_TO_BASE64[((buf_[1] & 0x0f) << 2) + ((buf_[2] & 0xc0) >> 6)];
  else        encoded[2]= INT_TO_BASE64[64];
  if(idx_>=2) encoded[3]= INT_TO_BASE64[buf_[2] & 0x3f];
  else        encoded[3]= INT_TO_BASE64[64];

  for(int i(0);i<4;++i,++(*out_stream_itr_))
    *(*out_stream_itr_)= encoded[i];

  idx_=0;
  written_= false;
}


template <typename t_istream_iterator>
class TInBase64Iterator
{
public:

  TInBase64Iterator(t_istream_iterator *is_itr)
    : in_stream_itr_(is_itr), in_stream_last(t_istream_iterator()), idx_(0), loaded_(0), is_eos_(false) {}

  TInBase64Iterator()
    : in_stream_itr_(NULL), in_stream_last(t_istream_iterator()), idx_(0), loaded_(0), is_eos_(false) {}

  ~TInBase64Iterator() {}

  char operator*()
    {
      if(loaded_==0)
      {
        load_block();
        if(loaded_==0)  return 0;
      }
      return buf_[idx_];
    }

  const TInBase64Iterator& operator++()
    {
      if(loaded_==0)
      {
        load_block();
        if(loaded_==0)  return *this;
      }
      ++idx_;
      if(idx_==loaded_)  {idx_=0;  loaded_=0;}
      return *this;
    }

  bool operator==(const TInBase64Iterator<t_istream_iterator> &rhs)
    {
      if(in_stream_itr_==NULL)
      {
        if(rhs.in_stream_itr_==NULL || rhs.IsEndOfStream())  return true;
        else  return false;
      }
      if(rhs.in_stream_itr_==NULL)
      {
        if(in_stream_itr_==NULL || IsEndOfStream())  return true;
        else  return false;
      }
      return (*in_stream_itr_)==(*rhs.in_stream_itr_) && idx_==rhs.idx_;
    }
  bool operator!=(const TInBase64Iterator<t_istream_iterator> &rhs)  {return !operator==(rhs);}

  bool IsEndOfStream() const {return is_eos_ && loaded_==0;}

private:

  t_istream_iterator *in_stream_itr_, in_stream_last;

  char buf_[3];
  int  idx_;
  int  loaded_;
  bool is_eos_;

  void load_block();

};

template <typename t_istream_iterator>
void TInBase64Iterator<t_istream_iterator>::load_block()
{
  // skip non-base64
  while(!IsBase64(*(*in_stream_itr_)))
  {
    ++(*in_stream_itr_);
    if((*in_stream_itr_) == in_stream_last)  {is_eos_=true; return;}
  }

  char base64[4];
  for(int i(0); i<4; ++i,++(*in_stream_itr_))
    base64[i]= Base64ToInt(*(*in_stream_itr_));

  buf_[0]= (base64[0] << 2) + ((base64[1] & 0x30) >> 4);
  if(base64[2]!=-1)
  {
    buf_[1]= ((base64[1] & 0x0f) << 4) + ((base64[2] & 0x3c) >> 2);
    if(base64[3]!=-1)
    {
      buf_[2]= ((base64[2] & 0x03) << 6) + base64[3];
      loaded_=3;
    }
    else
    {
      buf_[2]= 0;
      loaded_=2;
    }
  }
  else
  {
    buf_[1]= 0;
    loaded_=1;
  }

  if((*in_stream_itr_) == in_stream_last)  is_eos_=true;
}



//-------------------------------------------------------------------------------------------
}  // end of namespace loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // base64_test_h
//-------------------------------------------------------------------------------------------
