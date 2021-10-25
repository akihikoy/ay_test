//-------------------------------------------------------------------------------------------
/*! \file    sfm_simple1.cpp
    \brief   Simple structure from motion with optical flow LK where the camera 3D velocity is known.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.21, 2020

g++ -I -Wall -O2 sfm_simple1.cpp -o sfm_simple1.out -lopencv_core -lopencv_legacy -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <map>
#include "cap_open.h"
#define LIBRARY
#include "float_trackbar.cpp"
#include <sys/time.h>  // getrusage, gettimeofday
#include <sys/resource.h> // get cpu time
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
inline double GetUserTime(void)
{
  struct rusage RU;
  getrusage(RUSAGE_SELF, &RU);
  return static_cast<double>(RU.ru_utime.tv_sec) + static_cast<double>(RU.ru_utime.tv_usec)*1.0e-6;
}

inline cv::Scalar ValueToColGR(const float &value, const float &min, const float &max)
{
  cv::Scalar col(0.0,0.0,0.0);
  col[1]= (value<min) ? 255.0 : ( (value>max) ?   0.0 : (255.0-255.0/(max-min)*(value-min)) );
  col[2]= (value<min) ?   0.0 : ( (value>max) ? 255.0 : (255.0/(max-min)*(value-min)) );
  return col;
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
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  const char *window("Optical Flow LK");
  cv::namedWindow(window,1);
  int ni(1);
  float v_min(4.0), v_max(1000.0);
  CreateTrackbar<int>("Interval", window, &ni, 0, 100, 1,  NULL);
  CreateTrackbar<float>("v_min", window, &v_min, 0.0f, 100.0f, 0.1f,  NULL);
  CreateTrackbar<float>("v_max", window, &v_max, 0.0f, 1000.0f, 0.01f,  NULL);

  float vx_cam(0.0), vy_cam(-0.1);  // Camera 3D velocity (m/s).
  float Fx(200.0),Fy(200.0);       // Camera parameters.
  float viz_z_min(0.0), viz_z_max(0.1);    // Range of depth for visualization.

  CreateTrackbar<int>("Interval", window, &ni, 0, 100, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("vx_cam", window, &vx_cam, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("vy_cam", window, &vy_cam, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("viz_z_min", window, &viz_z_min, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("viz_z_max", window, &viz_z_max, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);

  cv::Mat frame_in, frame, frame_old;
  std::map<int,cv::Mat> history;
  std::map<int,float> history_tm;
  for(int i(0);;++i)
  {
    if(!cap.Read(frame_in))
    {
      if(cap.WaitReopen()) {i=-1; continue;}
      else break;
    }
    int N=100;
    cv::cvtColor(frame_in,frame,CV_BGR2GRAY);
    frame.copyTo(history[i]);
    history_tm[i]= GetUserTime();
    if(i>N)  {history.erase(i-N-1); history_tm.erase(i-N-1);}
    frame_old= history[((i-ni)>=0?(i-ni):0)];
    float dt= history_tm[i] - history_tm[((i-ni)>=0?(i-ni):0)];

    // medianBlur(frame, frame, 9);

    cv::Mat velx(frame.rows, frame.cols, CV_32FC1);
    cv::Mat vely(frame.rows, frame.cols, CV_32FC1);
    velx= cv::Scalar(0);
    vely= cv::Scalar(0);
    CvMat prev(frame_old), curr(frame), velx2(velx), vely2(vely);
    // Using HS:
    // CvTermCriteria criteria= cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 16, 0.1);
    // cvCalcOpticalFlowHS(&prev, &curr, 0, &velx2, &vely2, 10.0, criteria);
    // Using LK:
    cvCalcOpticalFlowLK(&prev, &curr, cv::Size(5,5), &velx2, &vely2);

    // Compute distance and visualize.
    frame_in*= 0.7;
    {
      cv::Scalar col;
      float vx,vy,spd,angle;
      int dx,dy;
      int step(1);
      float z_min(100.0), z_max(-100.0);
      int num_of(0);
      for (int i(step); i<frame_in.cols-step; i+=step)
      {
        for (int j(step); j<frame_in.rows-step; j+=step)
        {
          vx= velx.at<float>(j, i);  // Index order is y,x
          vy= vely.at<float>(j, i);  // Index order is y,x
          spd= std::sqrt(vx*vx+vy*vy);
          if(spd<v_min || v_max<spd)  continue;
          angle= std::atan2(vy,vx);
          ++num_of;

          float z= -((vx/Fx)*vx_cam+(vy/Fy)*vy_cam)*dt/((vx/Fx)*(vx/Fx)+(vy/Fy)*(vy/Fy));
          if(z<z_min)  z_min= z;
          if(z>z_max)  z_max= z;

          if(z>viz_z_min)
          {
            col= ValueToColGR(z, viz_z_min, viz_z_max);
            // cv::circle(frame_in, cv::Point(i,j), 1, col);
            frame_in.at<cv::Vec3b>(j,i)= cv::Vec3b(col[0],col[1],col[2]);
          }
        }
      }
      std::cout<<"z_min,z_max,num_of: "<<z_min<<"\t"<<z_max<<"\t"<<num_of<<std::endl;
    }

    cv::imshow(window, frame_in);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
