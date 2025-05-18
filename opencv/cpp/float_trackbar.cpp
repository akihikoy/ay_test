//-------------------------------------------------------------------------------------------
/*! \file    float_trackbar.cpp
    \brief   Extended trackbar where trackbars can be defined with min/max/step for int/float/double.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o float_trackbar.out float_trackbar.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <list>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// Extended trackbar class where trackbars can be defined with min/max/step for int/float/double/bool,
// and trackbars can be defined with std::vector<std::string> for std::string.
//-------------------------------------------------------------------------------------------
template<typename T>
int cvRoundTmpl(const T &val)  {return cvRound(val);}
template<> int cvRoundTmpl<unsigned short>(const unsigned short &val)  {return cvRound((int)val);}
template<> int cvRoundTmpl<unsigned int>(const unsigned int &val)  {return cvRound((int)val);}
template<> int cvRoundTmpl<unsigned long>(const unsigned long &val)  {return cvRound((int)val);}
template<typename T>
struct TExtendedTrackbarInfo;
template<typename T>
struct TExtendedTrackbarUtil
{
  typedef T TTrackValue;
  static T Convert(const TExtendedTrackbarInfo<T> &/*info*/, const TTrackValue &v)
    {
      return v;
    }
  static TTrackValue Invert(const TExtendedTrackbarInfo<T> &/*info*/, const T &v)
    {
      return v;
    }
};
template<typename T>
struct TExtendedTrackbarInfo
{
  typedef typename TExtendedTrackbarUtil<T>::TTrackValue TTrackValue;
  const std::string Name, WinName;
  int Position;
  int IntMax;
  T &Value;
  TTrackValue Min;
  TTrackValue Max;
  TTrackValue Step;
  typedef void (*TCallback)(const TExtendedTrackbarInfo<T>&, void*);
  TCallback OnUpdate;
  void *Reference;
  void *UserData;
  TExtendedTrackbarInfo(const std::string &name, const std::string &winname, T &value, const TTrackValue &min, const TTrackValue &max, const TTrackValue &step, TCallback on_update, void *user_data, void *reference)
    : Name(name), WinName(winname), Value(value), OnUpdate(NULL), Reference(NULL), UserData(NULL)
    {
      Min= min;
      Max= max;
      Step= step;
      IntMax= ToInt(Max);
      if(on_update)  OnUpdate= on_update;
      if(user_data)  UserData= user_data;
      if(reference)  Reference= reference;
      Position= ToInt(TExtendedTrackbarUtil<T>::Invert(*this, value));
    }
  T ToValue(int p) const
    {
      if(p>IntMax)  p= IntMax;
      if(p<0)  p= 0;
      return TExtendedTrackbarUtil<T>::Convert(*this, Min + Step*static_cast<TTrackValue>(p));
    }
  T ToValue() const
    {
      return ToValue(Position);
    }
  int ToInt(TTrackValue v) const
    {
      if(v>Max)  v= Max;
      if(v<Min)  v= Min;
      return cvRoundTmpl((v-Min)/Step);
    }
  void Update()
    {
      Value= ToValue();
      if(OnUpdate)  OnUpdate(*this, UserData);
    }
};
template<>
struct TExtendedTrackbarUtil<std::string>
{
  typedef int TTrackValue;
  static std::string Convert(const TExtendedTrackbarInfo<std::string> &info, const TTrackValue &v)
    {
      return (*reinterpret_cast<const std::vector<std::string>*>(info.Reference))[v];
    }
  static TTrackValue Invert(const TExtendedTrackbarInfo<std::string> &info, const std::string &v)
    {
      const std::vector<std::string> &ref(*reinterpret_cast<const std::vector<std::string>*>(info.Reference));
      std::vector<std::string>::const_iterator itr= std::find(ref.begin(),ref.end(),v);
      if(itr==ref.end())  return -1;
      return std::distance(ref.begin(), itr);
    }
};
std::list<TExtendedTrackbarInfo<float> > ExtendedTrackbarInfo_float;
std::list<TExtendedTrackbarInfo<double> > ExtendedTrackbarInfo_double;
std::list<TExtendedTrackbarInfo<short> > ExtendedTrackbarInfo_short;
std::list<TExtendedTrackbarInfo<unsigned short> > ExtendedTrackbarInfo_unsigned_short;
std::list<TExtendedTrackbarInfo<int> > ExtendedTrackbarInfo_int;
std::list<TExtendedTrackbarInfo<unsigned int> > ExtendedTrackbarInfo_unsigned_int;
std::list<TExtendedTrackbarInfo<long> > ExtendedTrackbarInfo_long;
std::list<TExtendedTrackbarInfo<unsigned long> > ExtendedTrackbarInfo_unsigned_long;
std::list<TExtendedTrackbarInfo<bool> > ExtendedTrackbarInfo_bool;
std::list<TExtendedTrackbarInfo<std::string> > ExtendedTrackbarInfo_string;
template<typename T>
std::list<TExtendedTrackbarInfo<T> >& ExtendedTrackbarInfo();
template<>
std::list<TExtendedTrackbarInfo<float> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_float;}
template<>
std::list<TExtendedTrackbarInfo<double> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_double;}
template<>
std::list<TExtendedTrackbarInfo<short> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_short;}
template<>
std::list<TExtendedTrackbarInfo<unsigned short> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_unsigned_short;}
template<>
std::list<TExtendedTrackbarInfo<int> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_int;}
template<>
std::list<TExtendedTrackbarInfo<unsigned int> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_unsigned_int;}
template<>
std::list<TExtendedTrackbarInfo<long> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_long;}
template<>
std::list<TExtendedTrackbarInfo<unsigned long> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_unsigned_long;}
template<>
std::list<TExtendedTrackbarInfo<bool> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_bool;}
template<>
std::list<TExtendedTrackbarInfo<std::string> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_string;}
template<typename T>
void ExtendedTrackbarOnChange(int,void *pi)
{
  TExtendedTrackbarInfo<T> &info(*reinterpret_cast<TExtendedTrackbarInfo<T>*>(pi));
  info.Update();
}
//-------------------------------------------------------------------------------------------
template<typename T>
int CreateTrackbarHelper(const std::string& trackbarname, const std::string& winname, T *value, const typename TExtendedTrackbarUtil<T>::TTrackValue &min, const typename TExtendedTrackbarUtil<T>::TTrackValue &max, const typename TExtendedTrackbarUtil<T>::TTrackValue &step, typename TExtendedTrackbarInfo<T>::TCallback on_track=NULL, void *user_data=NULL, void *reference=NULL)
{
  for(typename std::list<TExtendedTrackbarInfo<T> >::iterator itr(ExtendedTrackbarInfo<T>().begin()),itr_end(ExtendedTrackbarInfo<T>().end()); itr!=itr_end; ++itr)
  {
    if(itr->Name==trackbarname && itr->WinName==winname)
    {
      ExtendedTrackbarInfo<T>().erase(itr);
      break;
    }
  }
  ExtendedTrackbarInfo<T>().push_back(TExtendedTrackbarInfo<T>(trackbarname, winname, *value, min, max, step, on_track, user_data, reference));
  TExtendedTrackbarInfo<T> &pi(ExtendedTrackbarInfo<T>().back());
  return cv::createTrackbar(trackbarname, winname, &pi.Position, pi.IntMax, ExtendedTrackbarOnChange<T>, &pi);
}
template<typename T>
int CreateTrackbar(const std::string& trackbarname, const std::string& winname, T *value, const T &min, const T &max, const T &step, typename TExtendedTrackbarInfo<T>::TCallback on_track=NULL, void *user_data=NULL)
{
  return CreateTrackbarHelper<T>(trackbarname, winname, value, min, max, step, on_track, user_data);
}
template<typename T>
int CreateTrackbar(const std::string& trackbarname, const std::string& winname, T *value, TExtendedTrackbarInfo<bool>::TCallback on_track=NULL, void *user_data=NULL)
{
  return CreateTrackbarHelper<T>(trackbarname, winname, value, 0, 1, 1, on_track, user_data);
}
template<typename T>
int CreateTrackbar(const std::string& trackbarname, const std::string& winname, T *value, std::vector<std::string> &str_list, typename TExtendedTrackbarInfo<T>::TCallback on_track=NULL, void *user_data=NULL)
{
  return CreateTrackbarHelper<T>(trackbarname, winname, value, 0, str_list.size()-1, 1, on_track, user_data, &str_list);
}
//-------------------------------------------------------------------------------------------
template<typename T>
void TrackbarPrintOnTrack(const TExtendedTrackbarInfo<T> &info, void*)
{
  std::cerr<<info.Name<<"= "<<info.Value<<std::endl;
}
//-------------------------------------------------------------------------------------------


#ifndef LIBRARY
template<typename T>
void OnTrack(const TExtendedTrackbarInfo<T> &info, void*)
{
  std::cerr<<"Changed("<<info.Name<<"): "<<info.Value<<std::endl;
}

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);

  float b(1.0),g(1.0),r(1.0);
  int ksize(3);
  size_t step(10);
  bool negative(false);
  std::string channel("bgr");
  std::vector<std::string> channel_list;
  channel_list.push_back("b");
  channel_list.push_back("g");
  channel_list.push_back("r");
  channel_list.push_back("bgr");
  CreateTrackbar<float>("b", "camera", &b, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<float>("g", "camera", &g, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<float>("r", "camera", &r, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<int>("ksize", "camera", &ksize, 1, 15, 2, &TrackbarPrintOnTrack);
  CreateTrackbar<size_t>("step", "camera", &step, 1, 15, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("negative", "camera", &negative, &TrackbarPrintOnTrack);
  CreateTrackbar<std::string>("channel", "camera", &channel, channel_list, &TrackbarPrintOnTrack);
  std::cerr<<"# of trackbars(int/float/double/bool/string): "<<ExtendedTrackbarInfo<int>().size()<<", "<<ExtendedTrackbarInfo<float>().size()<<", "<<ExtendedTrackbarInfo<double>().size()<<", "<<ExtendedTrackbarInfo<bool>().size()<<", "<<ExtendedTrackbarInfo<std::string>().size()<<std::endl;

  cv::Mat frame, channels[3];
  for(int i(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    cv::split(frame, channels);
    if(ksize>1)
      for(int n(0);n<3;++n)
        cv::GaussianBlur(channels[n], channels[n], cv::Size(ksize,ksize), ksize, ksize);
    float ch[3]= {1.0,1.0,1.0};
    if(channel=="b")  ch[1]=ch[2]=0.0;
    else if(channel=="g")  ch[0]=ch[2]=0.0;
    else if(channel=="r")  ch[0]=ch[1]=0.0;
    if(!negative)
    {
      cv::Mat channels2[3]= {ch[0]*b*channels[0],ch[1]*g*channels[1],ch[2]*r*channels[2]};
      cv::merge(channels2, 3, frame);
    }
    else
    {
      cv::Mat channels2[3]= {ch[0]*(255-b*channels[0]),ch[1]*(255-g*channels[1]),ch[2]*(255-r*channels[2])};
      cv::merge(channels2, 3, frame);
    }
    cv::imshow("camera", frame);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    if(c=='r')
    {
      cv::destroyWindow("camera");
      cv::namedWindow("camera",1);
      CreateTrackbar<float>("b", "camera", &b, 0.0, 3.0, 0.001, &OnTrack);
      CreateTrackbar<float>("g", "camera", &g, 0.0, 3.0, 0.001, &OnTrack);
      CreateTrackbar<float>("r", "camera", &r, 0.0, 3.0, 0.001, &OnTrack);
      CreateTrackbar<int>("ksize", "camera", &ksize, 1, 15, 2, &TrackbarPrintOnTrack);
      CreateTrackbar<bool>("negative", "camera", &negative, &TrackbarPrintOnTrack);
      CreateTrackbar<std::string>("channel", "camera", &channel, channel_list, &TrackbarPrintOnTrack);
      std::cerr<<"# of trackbars(int/float/double/bool/string): "<<ExtendedTrackbarInfo<int>().size()<<", "<<ExtendedTrackbarInfo<float>().size()<<", "<<ExtendedTrackbarInfo<double>().size()<<", "<<ExtendedTrackbarInfo<bool>().size()<<", "<<ExtendedTrackbarInfo<std::string>().size()<<std::endl;
    }
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
