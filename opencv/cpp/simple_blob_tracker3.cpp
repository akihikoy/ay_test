//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_tracker3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.10, 2016

Very similar to simple_blob_tracker3.cpp
but we use thresholding to detect black markers.

g++ -g -Wall -O2 -o simple_blob_tracker3.out simple_blob_tracker3.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

Run:
  $ ./simple_blob_tracker3.out
  OR
  $ ./simple_blob_tracker3.out CAMERA_NUMBER
  CAMERA_NUMBER: Camera device number.
Usage:
  Press 'q' or Esc: Exit the program.
  Press 'c': Calibrate the tracker. Show a white paper or white wall during the calibration.
  Press 'W': On/off video capture.
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include "cv2-videoout2.h"
#include "rotate90n.h"
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
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
//-------------------------------------------------------------------------------------------
void read(const cv::FileNode &data, cv::Point2f &x, const cv::Point2f &default_value)
{
  #define PROC_VAR(v)  if(!data[#v].empty()) data[#v]>>x.v;
  PROC_VAR(x);
  PROC_VAR(y);
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------
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
//-------------------------------------------------------------------------------------------
void read(const cv::FileNode &data, cv::KeyPoint &x, const cv::KeyPoint &default_value)
{
  #define PROC_VAR(v)  if(!data[#v].empty()) data[#v]>>x.v;
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------
// void write(cv::FileStorage &fs, const std::string&, const cv::SimpleBlobDetector::Params &x)
// {
  // x.write(fs);
// }
// //-------------------------------------------------------------------------------------------
// void read(const cv::FileNode &data, cv::SimpleBlobDetector::Params &x, const cv::SimpleBlobDetector::Params &default_value)
// {
  // x.read(data);
// }
//-------------------------------------------------------------------------------------------
}  // namespace cv
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

inline float Dist(const cv::Point2f &p, const cv::Point2f &q)
{
  cv::Point2f d= p-q;
  return cv::sqrt(d.x*d.x + d.y*d.y);
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
    const float &ds_emp=4.0,  // Emphasize (scale) ratio of DS to draw
    const float &dp_emp=10.0  // Emphasize (scale) ratio of DP to draw
  );
// Track individual blobs.  prev: base, curr: current.
void TrackKeyPoints(
    const std::vector<cv::KeyPoint> &prev,
    const std::vector<cv::KeyPoint> &curr,
    std::vector<TPointMove> &move,
    const float &dist_min,  // Minimum distance change (i.e. sensitivity)
    const float &dist_max,  // Maximum distance change (too large might be noise)
    const float &ds_min,  // Minimum size change (i.e. sensitivity)
    const float &ds_max  // Maximum size change (too large might be noise)
  );
std::vector<cv::KeyPoint> CalibrateOrigin(
    const std::vector<std::vector<cv::KeyPoint> > &data,
    const float &dist_neighbor,  // Minimum distance to a neighbor blob.
    const float &dist_min,
    const float &dist_max,
    const float &ds_min,
    const float &ds_max
    // const float &dd_max
  );
//-------------------------------------------------------------------------------------------

struct TBlobTrackerParams
{
  // For blob detection:
  cv::SimpleBlobDetector::Params SBDParams;
  float DistNeighbor;  // Minimum distance to a neighbor blob.
  // For blob tracking:
  float DistMin;  // Minimum distance change (i.e. sensitivity)
  float DistMax;  // Maximum distance change (too large might be noise)
  float DSMin;  // Minimum size change (i.e. sensitivity)
  float DSMax;  // Maximum size change (too large might be noise)
  float DistMinCalib, DistMaxCalib, DSMinCalib, DSMaxCalib;  // Above parameters for calibration
  // For visualization:
  float DSEmp;  // Emphasize (scale) ratio of DS to draw
  float DPEmp;  // Emphasize (scale) ratio of DP to draw
  // For calibration:
  int NCalibPoints;  // Number of points for calibration

  TBlobTrackerParams();
};
void WriteToYAML(const std::vector<TBlobTrackerParams> &blob_params, const std::string &file_name);
void ReadFromYAML(std::vector<TBlobTrackerParams> &blob_params, const std::string &file_name);
//-------------------------------------------------------------------------------------------

class TBlobTracker
{
public:
  void Init();
  void Step(const cv::Mat &img);
  void Draw(cv::Mat &img);
  void Calibrate(cv::VideoCapture &cap, boost::function<void(cv::Mat&)> modifier=NULL);
  void Calibrate(const std::vector<cv::Mat> &images);

  void SaveCalib(const std::string &file_name) const;
  void LoadCalib(const std::string &file_name);

  TBlobTrackerParams& Params()  {return params_;}
  const TBlobTrackerParams& Params() const {return params_;}

  const std::vector<TPointMove>& Data() const {return keypoints_move_;}

private:
  TBlobTrackerParams params_;
  cv::Ptr<cv::SimpleBlobDetector> detector_;

  std::vector<cv::KeyPoint> keypoints_orig_, keypoints_curr_;
  std::vector<TPointMove> keypoints_move_;
};
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
// Implementation
//-------------------------------------------------------------------------------------------

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
//-------------------------------------------------------------------------------------------

void ReadFromYAML(std::vector<cv::KeyPoint> &keypoints, const std::string &file_name)
{
  keypoints.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["KeyPoints"];
  data>>keypoints;
  fs.release();
}
//-------------------------------------------------------------------------------------------


void DrawPointMoves(cv::Mat &img, const std::vector<TPointMove> &move,
    const cv::Scalar &col1, const cv::Scalar &col2,
    const float &ds_emp,  // Emphasize (scale) ratio of DS
    const float &dp_emp  // Emphasize (scale) ratio of DP
  )
{
  // for(std::vector<TPointMove>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
  // {
    // // cv::circle(img, m->Po, m->So, col1);
    // // cv::circle(img, m->Po, m->So+ds_emp*m->DS, col2, ds_emp*m->DS);
    // cv::circle(img, m->Po, m->So, col1, ds_emp*m->DS);
    // cv::line(img, m->Po, m->Po+dp_emp*m->DP, col2, 3);
  // }
  for(std::vector<TPointMove>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
    cv::circle(img, m->Po, m->So, col1, ds_emp*m->DS);
  for(std::vector<TPointMove>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
    cv::circle(img, m->Po+dp_emp*m->DP, m->So+ds_emp*m->DS, col2);
  for(std::vector<TPointMove>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
    cv::line(img, m->Po, m->Po+dp_emp*m->DP, col2, 3);
}
//-------------------------------------------------------------------------------------------

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
  // move.clear();
  // move.reserve(prev.size());
  move.resize(prev.size());
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
    // TPointMove m;
    TPointMove &m(move[p_idx]);
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
    // move.push_back(m);
  }
}
//-------------------------------------------------------------------------------------------

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
//-------------------------------------------------------------------------------------------

TBlobTrackerParams::TBlobTrackerParams()
{
  SBDParams.filterByColor= 0;
  SBDParams.blobColor= 0;
  // Change thresholds
  SBDParams.minThreshold = 5;
  SBDParams.maxThreshold = 200;
  // Filter by Area.
  SBDParams.filterByArea = true;
  SBDParams.minArea = 40;
  // Filter by Circularity
  SBDParams.filterByCircularity = true;
  SBDParams.minCircularity = 0.10;
  // Filter by Convexity
  SBDParams.filterByConvexity = true;
  SBDParams.minConvexity = 0.87;
  // Filter by Inertia
  SBDParams.filterByInertia = true;
  SBDParams.minInertiaRatio = 0.01;

  // For blob detection:
  DistNeighbor= 20.0;
  // For blob tracking:
  DistMin= 2.0;
  DistMax= 10.0;
  DSMin= 0.0;
  DSMax= 10.0;
  DistMinCalib= DistMin; DistMaxCalib= DistMax; DSMinCalib= DSMin; DSMaxCalib= DSMax;
  // For visualization:
  DSEmp= 4.0;
  DPEmp= 10.0;

  // For calibration:
  NCalibPoints= 20;
}
//-------------------------------------------------------------------------------------------

void WriteToYAML(const std::vector<TBlobTrackerParams> &blob_params, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs<<"BlobTracker"<<"[";
  for(std::vector<TBlobTrackerParams>::const_iterator itr(blob_params.begin()),itr_end(blob_params.end()); itr!=itr_end; ++itr)
  {
    fs<<"{";
    #define PROC_VAR(x,y)  fs<<#x"_"#y<<itr->x.y;
    PROC_VAR(SBDParams,filterByColor       );
    PROC_VAR(SBDParams,blobColor           );
    PROC_VAR(SBDParams,minThreshold        );
    PROC_VAR(SBDParams,maxThreshold        );
    PROC_VAR(SBDParams,filterByArea        );
    PROC_VAR(SBDParams,minArea             );
    PROC_VAR(SBDParams,filterByCircularity );
    PROC_VAR(SBDParams,minCircularity      );
    PROC_VAR(SBDParams,filterByConvexity   );
    PROC_VAR(SBDParams,minConvexity        );
    PROC_VAR(SBDParams,filterByInertia     );
    PROC_VAR(SBDParams,minInertiaRatio     );
    #undef PROC_VAR
    #define PROC_VAR(x)  fs<<#x<<itr->x;
    PROC_VAR(DistNeighbor );
    PROC_VAR(DistMin      );
    PROC_VAR(DistMax      );
    PROC_VAR(DSMin        );
    PROC_VAR(DSMax        );
    PROC_VAR(DistMinCalib );
    PROC_VAR(DistMaxCalib );
    PROC_VAR(DSMinCalib   );
    PROC_VAR(DSMaxCalib   );
    PROC_VAR(DSEmp        );
    PROC_VAR(DPEmp        );
    PROC_VAR(NCalibPoints );
    fs<<"}";
    #undef PROC_VAR
  }
  fs<<"]";
  fs.release();
}
//-------------------------------------------------------------------------------------------

void ReadFromYAML(std::vector<TBlobTrackerParams> &blob_params, const std::string &file_name)
{
  blob_params.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["BlobTracker"];
  for(cv::FileNodeIterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
  {
    TBlobTrackerParams cf;
    #define PROC_VAR(x,y)  if(!(*itr)[#x"_"#y].empty())  (*itr)[#x"_"#y]>>cf.x.y;
    PROC_VAR(SBDParams,filterByColor       );
    PROC_VAR(SBDParams,blobColor           );
    PROC_VAR(SBDParams,minThreshold        );
    PROC_VAR(SBDParams,maxThreshold        );
    PROC_VAR(SBDParams,filterByArea        );
    PROC_VAR(SBDParams,minArea             );
    PROC_VAR(SBDParams,filterByCircularity );
    PROC_VAR(SBDParams,minCircularity      );
    PROC_VAR(SBDParams,filterByConvexity   );
    PROC_VAR(SBDParams,minConvexity        );
    PROC_VAR(SBDParams,filterByInertia     );
    PROC_VAR(SBDParams,minInertiaRatio     );
    #undef PROC_VAR
    #define PROC_VAR(x)  if(!(*itr)[#x].empty())  (*itr)[#x]>>cf.x;
    PROC_VAR(DistNeighbor );
    PROC_VAR(DistMin      );
    PROC_VAR(DistMax      );
    PROC_VAR(DSMin        );
    PROC_VAR(DSMax        );
    PROC_VAR(DistMinCalib );
    PROC_VAR(DistMaxCalib );
    PROC_VAR(DSMinCalib   );
    PROC_VAR(DSMaxCalib   );
    PROC_VAR(DSEmp        );
    PROC_VAR(DPEmp        );
    PROC_VAR(NCalibPoints );
    #undef PROC_VAR
    blob_params.push_back(cf);
  }
  fs.release();
}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// class TBlobTracker
//-------------------------------------------------------------------------------------------

void TBlobTracker::Init()
{
  detector_= cv::SimpleBlobDetector::create(params_.SBDParams);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::Step(const cv::Mat &img)
{
  detector_->detect(img, keypoints_curr_);
  // cv::drawKeypoints(img, keypoints_curr_, img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  if(keypoints_orig_.size()==0)  keypoints_orig_= keypoints_curr_;
  TrackKeyPoints(keypoints_orig_, keypoints_curr_, keypoints_move_,
      params_.DistMin, params_.DistMax, params_.DSMin, params_.DSMax);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::Draw(cv::Mat &img)
{
  DrawPointMoves(img, keypoints_move_, cv::Scalar(255,0,0), cv::Scalar(0,0,255));
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::Calibrate(cv::VideoCapture &cap, boost::function<void(cv::Mat&)> modifier)
{
  std::cerr<<"Calibrating..."<<std::endl;
  // keypoints_orig_= keypoints_curr_;
  std::vector<std::vector<cv::KeyPoint> > data;
  cv::Mat frame;
  for(int i(0); i<params_.NCalibPoints; ++i)
  {
    cap >> frame; // get a new frame from camera
    if(modifier)  modifier(frame);
    detector_->detect(frame, keypoints_curr_);
    data.push_back(keypoints_curr_);
  }
  keypoints_orig_= CalibrateOrigin(data, params_.DistNeighbor,
      params_.DistMinCalib, params_.DistMaxCalib, params_.DSMinCalib, params_.DSMaxCalib);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::Calibrate(const std::vector<cv::Mat> &images)
{
  std::cerr<<"Calibrating..."<<std::endl;
  // keypoints_orig_= keypoints_curr_;
  std::vector<std::vector<cv::KeyPoint> > data;
  for(int i(0),i_end(images.size()); i<i_end; ++i)
  {
    detector_->detect(images[i], keypoints_curr_);
    data.push_back(keypoints_curr_);
  }
  keypoints_orig_= CalibrateOrigin(data, params_.DistNeighbor,
      params_.DistMinCalib, params_.DistMaxCalib, params_.DSMinCalib, params_.DSMaxCalib);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::SaveCalib(const std::string &file_name) const
{
  WriteToYAML(keypoints_orig_, file_name);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker::LoadCalib(const std::string &file_name)
{
  ReadFromYAML(keypoints_orig_, file_name);
}
//-------------------------------------------------------------------------------------------



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
  std::string cam("0");
  int n_rotate90(0);
  if(argc>1)  cam= argv[1];
  if(argc>2)  n_rotate90= atoi(argv[2]);

  cv::VideoCapture cap;
  cap= CapOpen(cam, /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;


  TBlobTracker tracker;
  tracker.Params().SBDParams.minArea= 10;
  tracker.Params().SBDParams.maxArea= 150;
  tracker.Params().DistMin= 0.0;
  tracker.Params().DistMax= 20.0;
  tracker.Params().DSMin= -1.0;
  tracker.Params().DSMax= 10.0;
  tracker.Params().DistMinCalib= 3.0;
  tracker.Params().DistMaxCalib= 20.0;
  tracker.Params().DSMinCalib= -1.0;
  tracker.Params().DSMaxCalib= 10.0;
  tracker.Init();

  std::string win("camera");
  cv::namedWindow(win,1);

  // int threshold_value1= 11;
  // cv::createTrackbar("threshold_value1", win, &threshold_value1, 255, NULL);
  // int n_erode1(0), n_dilate1(1);
  // cv::createTrackbar("n_dilate1", win, &n_dilate1, 10, NULL);
  // cv::createTrackbar("n_erode1", win, &n_erode1, 10, NULL);

  int thresh_h(180), thresh_s(255), thresh_v(13);
  cv::createTrackbar("thresh_h", win, &thresh_h, 255, NULL);
  cv::createTrackbar("thresh_s", win, &thresh_s, 255, NULL);
  cv::createTrackbar("thresh_v", win, &thresh_v, 255, NULL);
  int n_erode1(2), n_dilate1(2);
  cv::createTrackbar("n_dilate1", win, &n_dilate1, 10, NULL);
  cv::createTrackbar("n_erode1", win, &n_erode1, 10, NULL);

  TEasyVideoOut vout;
  vout.SetfilePrefix("/tmp/blobtr");

  cv::Mat frame;
  for(int f(0);;++f)
  {
    cap >> frame; // get a new frame from camera
    if(f%2!=0)  continue;  // Adjust FPS for speed up

    Rotate90N(frame,frame,n_rotate90);

    // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    // cv::threshold(frame, frame, threshold_value1, 255, cv::THRESH_TRUNC);
    // frame= frame*(255/(threshold_value1+1));
    // cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), n_dilate1);
    // cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), n_erode1);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    cv::inRange(frame, cv::Scalar(0, 0, 0), cv::Scalar(thresh_h, thresh_s, thresh_v), frame);
    cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), n_dilate1);
    cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), n_erode1);

    tracker.Step(frame);
    cv::Mat frames[3]= {frame,frame,frame}, framec;
    cv::merge(frames,3,framec); frame= framec;
    tracker.Draw(frame);

    vout.Step(frame);
    vout.VizRec(frame);
    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(char(c)=='W')  vout.Switch();
    else if(c==' ')  tracker.Calibrate(cap, boost::bind(Rotate90N,_1,_1,n_rotate90));
    else if(c=='c')
    {
      std::vector<cv::Mat> frames;
      for(int i(0); i<tracker.Params().NCalibPoints; ++i)
      {
        cap >> frame; // get a new frame from camera
        Rotate90N(frame,frame,n_rotate90);

        // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        // cv::threshold(frame, frame, threshold_value1, 255, cv::THRESH_TRUNC);
        // frame= frame*(255/(threshold_value1+1));
        // cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), n_dilate1);
        // cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), n_erode1);

        cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
        cv::inRange(frame, cv::Scalar(0, 0, 0), cv::Scalar(thresh_h, thresh_s, thresh_v), frame);
        cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), n_dilate1);
        cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), n_erode1);

        frames.push_back(frame.clone());
      }
      tracker.Calibrate(frames);
    }
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
