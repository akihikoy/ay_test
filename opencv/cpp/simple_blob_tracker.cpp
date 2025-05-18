//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_tracker.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.06, 2016

g++ -g -Wall -O2 -o simple_blob_tracker.out simple_blob_tracker.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cv2-videoout2.h"
#include "rotate90n.h"
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

bool ParamChanged(false);
void OnTrack(int,void*)
{
  ParamChanged= true;
}

int filterByColor = 0;
int minThreshold = 5;
int maxThreshold = 200;
int minArea = 40;
int minCircularity = 10;
int minConvexity = 87;
int minInertiaRatio = 1;

void AssignParams(cv::SimpleBlobDetector::Params &params)
{
  #define R100(x)  ((double)x*0.01)
  params.filterByColor= filterByColor;
  params.blobColor= 0;

  // Change thresholds
  params.minThreshold = minThreshold;
  params.maxThreshold = maxThreshold;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = minArea;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = R100(minCircularity);

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = R100(minConvexity);

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = R100(minInertiaRatio);
}

struct TPointMove
{
  cv::Point2f Po;  // Original position
  float So;        // Original size
  cv::Point2f DP;  // Displacement of position
  float DS;        // Displacement of size
};

void DrawPointMoves(cv::Mat &img, const std::vector<TPointMove> &move,
    const cv::Scalar &col1, const cv::Scalar &col2,
    const float &ds_emp=4.0,  // Emphasize (scale) ratio of DS
    const float &dp_emp=10.0  // Emphasize (scale) ratio of DP
  )
{
  for(std::vector<TPointMove>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
  {
    cv::circle(img, m->Po, m->So, col1);
    // cv::circle(img, m->Po, m->So+ds_emp*m->DS, col2, ds_emp*m->DS);
    cv::circle(img, m->Po, m->So, col2, ds_emp*m->DS);
    cv::line(img, m->Po, m->Po+dp_emp*m->DP, col2, 3);
  }
}

float Dist(const cv::Point2f &p, const cv::Point2f &q)
{
  cv::Point2f d= p-q;
  return cv::sqrt(d.x*d.x + d.y*d.y);
}

// Track individual blobs.  prev: base, curr: current.
void TrackKeyPoints(
    const std::vector<cv::KeyPoint> &prev,
    const std::vector<cv::KeyPoint> &curr,
    std::vector<TPointMove> &move,
    const float &dist_min,  // Minimum distance change (i.e. sensitivity)
    const float &dist_max,  // Maximum distance change (too large might be noise)
    const float &ds_min,  // Minimum size change (i.e. sensitivity)
    const float &ds_max  // Maximum size change (too large might be noise)
    // const float &dd_max   // Maximum change of distance change (low pass filter)
  )
{
  std::vector<TPointMove> old_move(move);  // for filter
  move.clear();
  move.reserve(prev.size());
  typedef std::pair<int,float> TTracked;
  std::vector<TTracked> tracked;  // [idx of curr]=(idx of prev, dist)
  tracked.resize(curr.size(), TTracked(-1,0.0));
  int p_idx(0);
  for(std::vector<cv::KeyPoint>::const_iterator p(prev.begin()),p_end(prev.end()); p!=p_end; ++p,++p_idx)
  {
    float dp_min(dist_max);
    std::vector<cv::KeyPoint>::const_iterator c_min(curr.end());
    int c_idx(0), c_min_idx(-1);
    for(std::vector<cv::KeyPoint>::const_iterator c(curr.begin()),c_end(curr.end()); c!=c_end; ++c,++c_idx)
    {
      float dist= Dist(p->pt, c->pt);
      if(dist<dp_min)
      {
        dp_min= dist;
        c_min= c;
        c_min_idx= c_idx;
      }
    }
    TPointMove m;
    m.Po= p->pt;
    m.So= p->size;
    cv::Point2f dp= c_min->pt - m.Po;
    float ds= c_min->size - m.So;
    float &dist(dp_min);  // norm of dp
    if(dist>=dist_min && dist<dist_max && ds>=ds_min && ds<ds_max
      && (tracked[c_min_idx].first<0 || tracked[c_min_idx].second<dist)
      /*&& (old_move.size()==0 || Dist(dp,old_move[p_idx].DP)<dd_max)*/ )
    {
      if(tracked[c_min_idx].first>=0)
      {
        move[tracked[c_min_idx].first].DP= cv::Point2f(0.0,0.0);
        move[tracked[c_min_idx].first].DS= 0.0;
      }
      m.DP= dp;
      m.DS= ds;
      tracked[c_min_idx]= TTracked(p_idx, dist);
    }
    else
    {
      m.DP= cv::Point2f(0.0,0.0);
      m.DS= 0.0;
    }
    move.push_back(m);
  }
}

std::vector<cv::KeyPoint> CalibrateOrigin(
    const std::vector<std::vector<cv::KeyPoint> > &data,
    const float &dist_neighbor,  // Minimum distance to a neighbor blob.
    const float &dist_min,
    const float &dist_max,
    const float &ds_min,
    const float &ds_max
    // const float &dd_max
  )
{
  std::vector<cv::KeyPoint> origin;
  if(data.size()==0)  return origin;
  std::vector<TPointMove> move;
  origin= data[0];
  for(int i(0),i_end(origin.size()); i<i_end; ++i)
  {
    for(int j(i_end-1); j>i; --j)
    {
      float dist= Dist(origin[i].pt, origin[j].pt);
      if(dist<dist_neighbor)
      {
        origin.erase(origin.begin()+j);
        --i_end;
      }
    }
  }
  for(int i(1),i_end(data.size()); i<i_end; ++i)
  {
    TrackKeyPoints(origin, data[i], move, dist_min, dist_max, ds_min, ds_max/*, dd_max*/);
    for(int j(move.size()-1); j>=0; --j)
    {
      if(move[j].DP.x!=0.0 || move[j].DP.y!=0.0)
        origin.erase(origin.begin()+j);
    }
  }
  return origin;
}

namespace cv
{
void write(cv::FileStorage &fs, const std::string&, const cv::Point2f &x)
{
  #define PROC_VAR(v)  fs<<#v<<x.v;
  fs<<"{";
  PROC_VAR(x);
  PROC_VAR(y);
  fs<<"}";
  #undef PROC_VAR
}
void read(const cv::FileNode &data, cv::Point2f &x, const cv::Point2f &default_value=cv::Point2f())
{
  #define PROC_VAR(v)  data[#v]>>x.v;
  PROC_VAR(x);
  PROC_VAR(y);
  #undef PROC_VAR
}
void write(cv::FileStorage &fs, const std::string&, const cv::KeyPoint &x)
{
  #define PROC_VAR(v)  fs<<#v<<x.v;
  fs<<"{";
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  fs<<"}";
  #undef PROC_VAR
}
void read(const cv::FileNode &data, cv::KeyPoint &x, const cv::KeyPoint &default_value=cv::KeyPoint())
{
  #define PROC_VAR(v)  data[#v]>>x.v;
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  #undef PROC_VAR
}
}

void WriteToYAML(const std::vector<cv::KeyPoint> &keypoints, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  // fs<<"KeyPoints"<<keypoints;
  fs<<"KeyPoints"<<"[";
  for(std::vector<cv::KeyPoint>::const_iterator itr(keypoints.begin()),itr_end(keypoints.end()); itr!=itr_end; ++itr)
  {
    fs<<*itr;
  }
  fs<<"]";
  fs.release();
}

void ReadFromYAML(std::vector<cv::KeyPoint> &keypoints, const std::string &file_name)
{
  keypoints.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["KeyPoints"];
  data>>keypoints;
  fs.release();
}


int main(int argc, char**argv)
{
  int cam(0), n_rotate90(0);
  std::string keypoint_file;
  if(argc>1)  cam= atoi(argv[1]);
  if(argc>2)  n_rotate90= atoi(argv[2]);
  if(argc>3)  keypoint_file= argv[3];

  cv::VideoCapture cap(cam);
  cap.release();
  cap.open(cam);
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);


  // Setup SimpleBlobDetector parameters.
  cv::SimpleBlobDetector::Params params;
  AssignParams(params);

  // Set up the detector with default parameters.
  cv::Ptr<cv::SimpleBlobDetector> detector;
  detector= cv::SimpleBlobDetector::create(params);

  // Detect blobs.
  std::vector<cv::KeyPoint> keypoints_orig, keypoints_curr;
  std::vector<TPointMove> keypoints_move;

  if(keypoint_file!="")
  {
    std::cerr<<"Loading keypoints from: "<<keypoint_file<<std::endl;
    ReadFromYAML(keypoints_orig, keypoint_file);
    std::cerr<<"Read points: "<<keypoints_orig.size()<<std::endl;
    /*DEBUG*/WriteToYAML(keypoints_orig, "/tmp/blob_keypoints.yaml");
  }

  std::string win("camera");
  cv::namedWindow(win,1);

  // cv::createTrackbar("filterByColor", win, &filterByColor, 1, OnTrack);
  // cv::createTrackbar("minThreshold", win, &minThreshold, 255, OnTrack);
  // cv::createTrackbar("maxThreshold", win, &maxThreshold, 255, OnTrack);
  // cv::createTrackbar("minArea", win, &minArea, 5000, OnTrack);
  // cv::createTrackbar("minCircularity", win, &minCircularity, 100, OnTrack);
  // cv::createTrackbar("minConvexity", win, &minConvexity, 100, OnTrack);
  // cv::createTrackbar("minInertiaRatio", win, &minInertiaRatio, 100, OnTrack);

  int dist_neighbor(20);
  int dist_min(2), dist_max(10), ds_min(0), ds_max(10)/*, dd_max(5)*/;
  cv::createTrackbar("dist_min", win, &dist_min, 10, NULL);
  cv::createTrackbar("dist_max", win, &dist_max, 50, NULL);
  cv::createTrackbar("ds_min", win, &ds_min, 10, NULL);
  cv::createTrackbar("ds_max", win, &ds_max, 50, NULL);

  TEasyVideoOut vout;
  vout.SetfilePrefix("/tmp/blobtr");

  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    Rotate90N(frame,frame,n_rotate90);

    if(ParamChanged)
    {
      AssignParams(params);
      detector= cv::SimpleBlobDetector::create(params);
      ParamChanged= false;
    }

    detector->detect(frame, keypoints_curr);
    // cv::drawKeypoints(frame, keypoints_curr, frame, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if(keypoints_orig.size()==0)  keypoints_orig= keypoints_curr;
    TrackKeyPoints(keypoints_orig, keypoints_curr, keypoints_move,
        /*dist_min=*/dist_min, /*dist_max=*/dist_max, /*ds_min=*/ds_min, /*ds_max=*/ds_max);
    DrawPointMoves(frame, keypoints_move, cv::Scalar(255,0,0), cv::Scalar(0,0,255));

    vout.Step(frame);
    vout.VizRec(frame);
    cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(char(c)=='W')  vout.Switch();
    else if(c==' ')
    {
      std::cerr<<"Calibrating..."<<std::endl;
      // keypoints_orig= keypoints_curr;
      std::vector<std::vector<cv::KeyPoint> > data;
      for(int i(0); i<60; ++i)
      {
        cap >> frame; // get a new frame from camera
        Rotate90N(frame,frame,n_rotate90);
        detector->detect(frame, keypoints_curr);
        data.push_back(keypoints_curr);
      }
      keypoints_orig= CalibrateOrigin(data, dist_neighbor, dist_min, dist_max, ds_min, ds_max);
      WriteToYAML(keypoints_orig, "data/blob_keypoints.yaml");
    }
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
