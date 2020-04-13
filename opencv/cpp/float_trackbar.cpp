//-------------------------------------------------------------------------------------------
/*! \file    float_trackbar.cpp
    \brief   Trackbar for floating point values;
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o float_trackbar.out float_trackbar.cpp -lopencv_core -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <list>
#include "cap_open.h"
#include <unistd.h>  // usleep
//-------------------------------------------------------------------------------------------

template<typename T>
struct TFloatTrackbarInfo
{
  const std::string &Name;
  int Position;
  int IntMax;
  T &Value;
  T Min;
  T Max;
  T Step;
  typedef void (*TCallback)(const TFloatTrackbarInfo<T>&, void*);
  TCallback OnUpdate;
  void *UserData;
  TFloatTrackbarInfo(const std::string &name, T &value, const T &min, const T &max, const T &step, TCallback on_update, void *user_data)
    : Name(name), Value(value), OnUpdate(NULL), UserData(NULL)
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
std::list<TFloatTrackbarInfo<float> > FloatTrackbarInfo_float;
std::list<TFloatTrackbarInfo<double> > FloatTrackbarInfo_double;
template<typename T>
std::list<TFloatTrackbarInfo<T> >& FloatTrackbarInfo();
template<>
std::list<TFloatTrackbarInfo<float> >& FloatTrackbarInfo()  {return FloatTrackbarInfo_float;}
template<>
std::list<TFloatTrackbarInfo<double> >& FloatTrackbarInfo()  {return FloatTrackbarInfo_double;}

template<typename T>
void FloatTrackbarOnChange(int,void *pi)
{
  TFloatTrackbarInfo<T> &info(*reinterpret_cast<TFloatTrackbarInfo<T>*>(pi));
  info.Update();
}

template<typename T>
int CreateFloatTrackbar(const std::string& trackbarname, const std::string& winname, T *value, const T &min, const T &max, const T &step, typename TFloatTrackbarInfo<T>::TCallback on_track=NULL, void *user_data=NULL)
{
  FloatTrackbarInfo<T>().push_back(TFloatTrackbarInfo<T>(trackbarname, *value, min, max, step, on_track, user_data));
  TFloatTrackbarInfo<T> &pi(FloatTrackbarInfo<T>().back());
  return cv::createTrackbar(trackbarname, winname, &pi.Position, pi.IntMax, FloatTrackbarOnChange<T>, &pi);
}

template<typename T>
void FloatTrackbarPrintOnTrack(const TFloatTrackbarInfo<T> &info, void*)
{
  std::cerr<<info.Name<<"= "<<info.Value<<std::endl;
}


#ifndef LIBRARY
template<typename T>
void OnTrack(const TFloatTrackbarInfo<T> &info, void*)
{
  std::cerr<<"Moved: "<<info.Value<<std::endl;
}

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);

  float sleep_per_frame(0.1);
  CreateFloatTrackbar<float>("sleep_per_frame", "camera", &sleep_per_frame, 0.0, 0.5, 0.001, &OnTrack);

  cv::Mat frame;
  for(int i(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    cv::imshow("camera", frame);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    usleep(sleep_per_frame*1.0e6);
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
