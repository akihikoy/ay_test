//-------------------------------------------------------------------------------------------
/*! \file    rate_adjust3.cpp
    \brief   rate adjuster v.3
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.29, 2022

    g++ rate_adjust3.cpp
    ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <sys/time.h>
#include <chrono>
#include <thread>

#ifdef __linux__
  #include <inttypes.h>  // int64_t
  #include <sys/time.h>  // gettimeofday
#elif _WIN32
  #include <stdint.h>
  #define NOMINMAX  // Disabling min and max macros in Windows.h
  #include <Windows.h>
  #undef GetCurrentTime  // Remove the macro defined in Windows.h
#else
  #error OS detection failed
#endif

inline double GetCurrentTime(void)
{
#ifdef __linux__
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
#elif _WIN32
  ULONGLONG UnbiasedInterruptTime;
  QueryUnbiasedInterruptTime(&UnbiasedInterruptTime);
  return (double)UnbiasedInterruptTime/(double)10000000;
#else
  #error OS detection failed
#endif
}
//-------------------------------------------------------------------------------------------

class TRateAdjuster
{
public:
  typedef double Duration;
  typedef double Time;
  TRateAdjuster(const double &frequency)
      : expected_next_start_(GetCurrentTime()),
        actual_prev_start_(GetCurrentTime()),
        expected_cycle_time_(1.0 / frequency),
        actual_cycle_time_(0.0)
    {}
  void Sleep()
    {
      const Time &expected_curr_start(expected_next_start_);
      Time expected_next_end= expected_curr_start + expected_cycle_time_;
      Time actual_prev_end= GetCurrentTime();
      if(actual_prev_end < expected_curr_start)  expected_next_end= actual_prev_end+expected_cycle_time_;
      Duration sleep_time= expected_next_end - actual_prev_end;
      actual_cycle_time_= actual_prev_end-actual_prev_start_;
      actual_prev_start_= actual_prev_end;
      expected_next_start_= expected_next_end;
      if(sleep_time <= Duration(0.0))
      {
        if(actual_prev_end > expected_next_end+expected_cycle_time_)  expected_next_start_= actual_prev_end;
        return;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(int(sleep_time*1.0e9)));
    }
  void Reset()  {expected_next_start_= GetCurrentTime();}
  Duration ActualCycleTime() const {return actual_cycle_time_;}
  Duration ExpectedCycleTime() const {return expected_cycle_time_;}

private:
  Time expected_next_start_, actual_prev_start_;
  Duration expected_cycle_time_, actual_cycle_time_;
};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
#include <iostream>
using namespace std;

int main(int argc, char**argv)
{
  TRateAdjuster rate_adjuster(5);

  for(int i(0);;++i)
  {
    cout<<i<<" "<<rate_adjuster.ActualCycleTime()<<" "<<rate_adjuster.ExpectedCycleTime()<<endl;
    rate_adjuster.Sleep();
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
