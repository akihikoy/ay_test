//-------------------------------------------------------------------------------------------
/*! \file    rate_adjust.cpp
    \brief   rate adjuster
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.11, 2018

    g++ rate_adjust.cpp
    ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <sys/time.h>
#include <unistd.h>

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}

typedef double Duration;
typedef double Time;

// The code of the class Rate is from
// http://docs.ros.org/diamondback/api/rostime/html/classros_1_1Rate.html

class Rate
{
public:
  Rate(double frequency);

  bool sleep();

  void reset();

  Duration cycleTime();

  Duration expectedCycleTime() { return expected_cycle_time_; }

private:
  Time start_;
  Duration expected_cycle_time_, actual_cycle_time_;
};

Rate::Rate(double frequency)
: start_(GetCurrentTime())
, expected_cycle_time_(1.0 / frequency)
, actual_cycle_time_(0.0)
{}

bool Rate::sleep()
{
  Time expected_end = start_ + expected_cycle_time_;

  Time actual_end = GetCurrentTime();

  // detect backward jumps in time
  if (actual_end < start_)
  {
    expected_end = actual_end + expected_cycle_time_;
  }

  //calculate the time we'll sleep for
  Duration sleep_time = expected_end - actual_end;

  //set the actual amount of time the loop took in case the user wants to know
  actual_cycle_time_ = actual_end - start_;

  //make sure to reset our start time
  start_ = expected_end;

  //if we've taken too much time we won't sleep
  if(sleep_time <= Duration(0.0))
  {
    // if we've jumped forward in time, or the loop has taken more than a full extra
    // cycle, reset our cycle
    if (actual_end > expected_end + expected_cycle_time_)
    {
      start_ = actual_end;
    }
    return true;
  }

  return usleep(sleep_time*1.0e6);
}

void Rate::reset()
{
  start_ = GetCurrentTime();
}

Duration Rate::cycleTime()
{
  return actual_cycle_time_;
}


//-------------------------------------------------------------------------------------------
#include <iostream>
using namespace std;

int main(int argc, char**argv)
{
  Rate rate(5);

  for(int i(0);;++i)
  {
    cout<<i<<" "<<rate.cycleTime()<<endl;
    rate.sleep();
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
