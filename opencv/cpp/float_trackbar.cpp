//-------------------------------------------------------------------------------------------
/*! \file    float_trackbar.cpp
    \brief   Extended trackbar where trackbars can be defined with min/max/step for int/float/double.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o float_trackbar.out float_trackbar.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
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
// Extended trackbar where trackbars can be defined with min/max/step for int/float/double.
//-------------------------------------------------------------------------------------------
template<typename T>
struct TExtendedTrackbarInfo
{
  const std::string Name, WinName;
  int Position;
  int IntMax;
  T &Value;
  T Min;
  T Max;
  T Step;
  typedef void (*TCallback)(const TExtendedTrackbarInfo<T>&, void*);
  TCallback OnUpdate;
  void *UserData;
  TExtendedTrackbarInfo(const std::string &name, const std::string &winname, T &value, const T &min, const T &max, const T &step, TCallback on_update, void *user_data)
    : Name(name), WinName(winname), Value(value), OnUpdate(NULL), UserData(NULL)
    {
      Min= min;
      Max= max;
      Step= step;
      Position= ToInt(value);
      IntMax= ToInt(Max);
      if(on_update)  OnUpdate= on_update;
      if(user_data)  UserData= user_data;
    }
  T ToFloat(int p) const
    {
      if(p>IntMax)  p= IntMax;
      if(p<0)  p= 0;
      return Min + Step*static_cast<T>(p);
    }
  T ToFloat() const
    {
      return ToFloat(Position);
    }
  int ToInt(T v) const
    {
      if(v>Max)  v= Max;
      if(v<Min)  v= Min;
      return cvRound((v-Min)/Step);
    }
  void Update()
    {
      Value= ToFloat();
      if(OnUpdate)  OnUpdate(*this, UserData);
    }
};
std::list<TExtendedTrackbarInfo<float> > ExtendedTrackbarInfo_float;
std::list<TExtendedTrackbarInfo<double> > ExtendedTrackbarInfo_double;
std::list<TExtendedTrackbarInfo<int> > ExtendedTrackbarInfo_int;
std::list<TExtendedTrackbarInfo<bool> > ExtendedTrackbarInfo_bool;
template<typename T>
std::list<TExtendedTrackbarInfo<T> >& ExtendedTrackbarInfo();
template<>
std::list<TExtendedTrackbarInfo<float> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_float;}
template<>
std::list<TExtendedTrackbarInfo<double> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_double;}
template<>
std::list<TExtendedTrackbarInfo<int> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_int;}
template<>
std::list<TExtendedTrackbarInfo<bool> >& ExtendedTrackbarInfo()  {return ExtendedTrackbarInfo_bool;}
template<typename T>
void ExtendedTrackbarOnChange(int,void *pi)
{
  TExtendedTrackbarInfo<T> &info(*reinterpret_cast<TExtendedTrackbarInfo<T>*>(pi));
  info.Update();
}
//-------------------------------------------------------------------------------------------
template<typename T>
int CreateTrackbar(const std::string& trackbarname, const std::string& winname, T *value, const T &min, const T &max, const T &step, typename TExtendedTrackbarInfo<T>::TCallback on_track=NULL, void *user_data=NULL)
{
  for(typename std::list<TExtendedTrackbarInfo<T> >::iterator itr(ExtendedTrackbarInfo<T>().begin()),itr_end(ExtendedTrackbarInfo<T>().end()); itr!=itr_end; ++itr)
  {
    if(itr->Name==trackbarname && itr->WinName==winname)
    {
      ExtendedTrackbarInfo<T>().erase(itr);
      break;
    }
  }
  ExtendedTrackbarInfo<T>().push_back(TExtendedTrackbarInfo<T>(trackbarname, winname, *value, min, max, step, on_track, user_data));
  TExtendedTrackbarInfo<T> &pi(ExtendedTrackbarInfo<T>().back());
  return cv::createTrackbar(trackbarname, winname, &pi.Position, pi.IntMax, ExtendedTrackbarOnChange<T>, &pi);
}
template<typename T>
int CreateTrackbar(const std::string& trackbarname, const std::string& winname, T *value, TExtendedTrackbarInfo<bool>::TCallback on_track=NULL, void *user_data=NULL)
{
  return CreateTrackbar<T>(trackbarname, winname, value, 0, 1, 1, on_track, user_data);
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
  bool negative(false);
  CreateTrackbar<float>("b", "camera", &b, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<float>("g", "camera", &g, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<float>("r", "camera", &r, 0.0, 2.0, 0.001, &OnTrack);
  CreateTrackbar<int>("ksize", "camera", &ksize, 1, 9, 2, &TrackbarPrintOnTrack);
  CreateTrackbar<bool>("negative", "camera", &negative, &TrackbarPrintOnTrack);
  std::cerr<<"# of trackbars(int/float/double/bool): "<<ExtendedTrackbarInfo<int>().size()<<", "<<ExtendedTrackbarInfo<float>().size()<<", "<<ExtendedTrackbarInfo<double>().size()<<", "<<ExtendedTrackbarInfo<bool>().size()<<std::endl;

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
    if(!negative)
    {
      cv::Mat channels2[3]= {b*channels[0],g*channels[1],r*channels[2]};
      cv::merge(channels2, 3, frame);
    }
    else
    {
      cv::Mat channels2[3]= {255-b*channels[0],255-g*channels[1],255-r*channels[2]};
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
      std::cerr<<"# of trackbars(int/float/double/bool): "<<ExtendedTrackbarInfo<int>().size()<<", "<<ExtendedTrackbarInfo<float>().size()<<", "<<ExtendedTrackbarInfo<double>().size()<<", "<<ExtendedTrackbarInfo<bool>().size()<<std::endl;
    }
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
