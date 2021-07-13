//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_tracker4b.cpp
    \brief   Blob tracker test code back ported from the FV project.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.14, 2021

g++ -g -Wall -O2 -o simple_blob_tracker4b.out simple_blob_tracker4b.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui

Run:
  $ ./simple_blob_tracker4b.out
  OR
  $ ./simple_blob_tracker4b.out CAMERA_NUMBER
  CAMERA_NUMBER: Camera device number.
Usage:
  Press 'q' or Esc: Exit the program.
  Press 'c': Calibrate the tracker (detecting markers and storing the initial positions and sizes).
             Tips: Show a white paper or white wall during the calibration.
  Press 'C': Show/hide the parameter configuration trackbars.
  Press 'W' (shift+'w'): Start/stop video recording.
  Press 'p': Print the calibration result.
  Press 's': Save the calibration result to file "blob_calib.yaml".
  Press 'l': Load a calibration from file "blob_calib.yaml".
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
#define LIBRARY
#include "float_trackbar.cpp"
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
namespace trick
{

inline float Dist(const cv::Point2f &p, const cv::Point2f &q)
{
  cv::Point2f d= p-q;
  return cv::sqrt(d.x*d.x + d.y*d.y);
}
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

//-------------------------------------------------------------------------------------------
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
/*! blob_tracker2.h */
//-------------------------------------------------------------------------------------------
#ifndef blob_tracker2_h
#define blob_tracker2_h
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <boost/function.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

struct TPointMove2
{
  cv::Point2f Po;  // Original position
  float So;        // Original size
  cv::Point2f DP;  // Displacement of position
  float DS;        // Displacement of size
  int NTrackFailed; // Number of traking failures in a raw
};
void DrawPointMoves2(cv::Mat &img, const std::vector<TPointMove2> &move,
    const cv::Scalar &col1, const cv::Scalar &col2,
    const float &ds_emp,  // Emphasize (scale) ratio of DS
    const float &dp_emp,  // Emphasize (scale) ratio of DP
    const cv::Mat &img_th,  // Processed image
    bool is_thresholded,  // If img_th is thresholded
    const float &s_width  // Width of search ROI of each keypoint
  );
// Track individual blobs.
void TrackKeyPoints2(
    const cv::Mat &img_th,  // Preprocessed image
    bool is_thresholded,   // If img_th is thresholded
    const cv::SimpleBlobDetector &detector,  // Blob detector
    const std::vector<cv::KeyPoint> &orig,  // Original keypoints
    std::vector<TPointMove2> &move,  // Must be previous movement
    const float &s_width,  // Width of search ROI of each keypoint
    const float &nonzero_min,  // Minimum ratio of nonzero pixels in ROI over original keypoint size.
    const float &nonzero_max,  // Maximum ratio of nonzero pixels in ROI over original keypoint size.
    const float &vp_max,  // Maximum position change (too large one might be noise)
    const float &vs_max,  // Maximum size change (too large one might be noise)
    const int   &n_reset  // When number of tracking failure in a row exceeds this, tracking is reset
  );
//-------------------------------------------------------------------------------------------

struct TBlobTracker2Params
{
  // For blob detection:
  cv::SimpleBlobDetector::Params SBDParams;

  // For preprocessing:
  bool ThresholdingImg;  // Thresholding the input image where Thresh*, NDilate1, NErode1 are used.
  int ThreshH;
  int ThreshS;
  int ThreshV;
  int NDilate1;
  int NErode1;
  // For keypoint tracking;
  float SWidth;  // Width of search ROI of each keypoint
  float NonZeroMin; // Minimum ratio of nonzero pixels in ROI over original keypoint size (used only when ThresholdingImg==true).
  float NonZeroMax; // Maximum ratio of nonzero pixels in ROI over original keypoint size (used only when ThresholdingImg==true).
  float VPMax;  // Maximum position change (too large one might be noise)
  float VSMax;  // Maximum size change (too large one might be noise)
  int   NReset;  // When number of tracking failure in a row exceeds this, tracking is reset
  // For calibration:
  float DistMaxCalib;
  float DSMaxCalib;

  // For visualization:
  float DSEmp;  // Emphasize (scale) ratio of DS to draw
  float DPEmp;  // Emphasize (scale) ratio of DP to draw
  // For calibration:
  int NCalibPoints;  // Number of points for calibration

  TBlobTracker2Params();
};
void WriteToYAML(const std::vector<TBlobTracker2Params> &blob_params, const std::string &file_name);
void ReadFromYAML(std::vector<TBlobTracker2Params> &blob_params, const std::string &file_name);
//-------------------------------------------------------------------------------------------

class TBlobTracker2
{
public:
  // User defined identifier.
  std::string Name;

  void Init();
  void Preprocess(const cv::Mat &img, cv::Mat &img_th);
  void Step(const cv::Mat &img);
  void Draw(cv::Mat &img);
  void Calibrate(const std::vector<cv::Mat> &images);

  // Remove a keypoint around p.
  void RemovePointAt(const cv::Point2f &p, const float &max_dist=30.0);

  void SaveCalib(const std::string &file_name) const;
  void LoadCalib(const std::string &file_name);

  TBlobTracker2Params& Params()  {return params_;}
  const TBlobTracker2Params& Params() const {return params_;}

  const std::vector<TPointMove2>& Data() const {return keypoints_move_;}

private:
  TBlobTracker2Params params_;
  cv::Ptr<cv::SimpleBlobDetector> detector_;

  std::vector<cv::KeyPoint> keypoints_orig_;
  std::vector<TPointMove2> keypoints_move_;

  cv::Mat img_th_;
};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // blob_tracker2_h
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
/*! blob_tracker2.cpp */
//-------------------------------------------------------------------------------------------
#include <opencv2/imgproc/imgproc.hpp>
//-------------------------------------------------------------------------------------------
namespace trick
{
using namespace std;
// using namespace boost;


void DrawPointMoves2(cv::Mat &img, const std::vector<TPointMove2> &move,
    const cv::Scalar &col1, const cv::Scalar &col2,
    const float &ds_emp,  // Emphasize (scale) ratio of DS
    const float &dp_emp,  // Emphasize (scale) ratio of DP
    const cv::Mat &img_th,  // Processed image
    bool is_thresholded,  // If img_th is thresholded
    const float &s_width  // Width of search ROI of each keypoint
  )
{
  // Debug mode:
  if(is_thresholded)
  {
    if(img_th.cols*img_th.rows>0)
    {
      // img*= 0.4;
      cv::Mat img_ths[3]= {0.4*img_th,0.2*img_th,0.0*img_th}, img_thc;
      cv::merge(img_ths,3,img_thc);
      img+= img_thc;
    }
  }

  // Draw ROIs
  for(int i(0),i_end(move.size()); i<i_end; ++i)
  {
    const TPointMove2 &mi(move[i]);
    cv::Point2f pc= mi.Po + mi.DP;  // Current marker position.
    cv::Rect roi(pc.x-s_width*0.5, pc.y-s_width*0.5, s_width, s_width);
    if(roi.x<0)  {roi.width+= roi.x; roi.x= 0;}
    if(roi.y<0)  {roi.height+= roi.y; roi.y= 0;}
    if(roi.width<=0 || roi.height<=0)  continue;
    if(roi.x+roi.width>img_th.cols)  {roi.width= img_th.cols-roi.x;}
    if(roi.y+roi.height>img_th.rows)  {roi.height= img_th.rows-roi.y;}
    if(roi.width<=0 || roi.height<=0)  continue;
    cv::rectangle(img, roi, col2, 1);
  }

  // for(std::vector<TPointMove2>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
  // {
    // // cv::circle(img, m->Po, m->So, col1);
    // // cv::circle(img, m->Po, m->So+ds_emp*m->DS, col2, ds_emp*m->DS);
    // cv::circle(img, m->Po, m->So, col1, ds_emp*m->DS);
    // cv::line(img, m->Po, m->Po+dp_emp*m->DP, col2, 3);
  // }
  int i(0);
  for(std::vector<TPointMove2>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m,++i)
  {
    cv::circle(img, m->Po, std::max(0.0f,m->So), col1, 1/*std::max(0.0f,ds_emp*m->DS)*/);
    std::stringstream ss; ss<<i;
    cv::putText(img, ss.str(), m->Po+cv::Point2f(3+std::max(0.0f,m->So),0), cv::FONT_HERSHEY_SIMPLEX, 0.3, col1, 1, CV_AA);
  }
  for(std::vector<TPointMove2>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
    cv::line(img, m->Po, m->Po+dp_emp*m->DP, col2, 3);
  for(std::vector<TPointMove2>::const_iterator m(move.begin()),m_end(move.end()); m!=m_end; ++m)
    cv::circle(img, m->Po+dp_emp*m->DP, std::max(0.0f,m->So+ds_emp*m->DS), col2, std::max(0.0f,ds_emp*m->DS));
}
//-------------------------------------------------------------------------------------------

void InitKeyPointMove(const std::vector<cv::KeyPoint> &orig, std::vector<TPointMove2> &move)
{
  move.resize(orig.size());
  for(int i(0),i_end(move.size()); i<i_end; ++i)
  {
    TPointMove2 &mi(move[i]);
    mi.Po= orig[i].pt;
    mi.So= orig[i].size;
    mi.DP= cv::Point2f(0.0,0.0);
    mi.DS= 0.0;
    mi.NTrackFailed= 0;
  }
}
//-------------------------------------------------------------------------------------------

// Track individual blobs.
void TrackKeyPoints2(
    const cv::Mat &img_th,  // Preprocessed image
    bool is_thresholded,   // If img_th is thresholded
    const cv::SimpleBlobDetector &detector,  // Blob detector
    const std::vector<cv::KeyPoint> &orig,  // Original keypoints
    std::vector<TPointMove2> &move,  // Must be previous movement
    const float &s_width,  // Width of search ROI of each keypoint
    const float &nonzero_min,  // Minimum ratio of nonzero pixels in ROI over original keypoint size.
    const float &nonzero_max,  // Maximum ratio of nonzero pixels in ROI over original keypoint size.
    const float &vp_max,  // Maximum position change (too large one might be noise)
    const float &vs_max,  // Maximum size change (too large one might be noise)
    const int   &n_reset  // When number of tracking failure in a row exceeds this, tracking is reset
  )
{
  assert(orig.size()==move.size());

  std::vector<cv::KeyPoint> keypoints;
  for(int i(0),i_end(orig.size()); i<i_end; ++i)
  {
    const cv::KeyPoint &oi(orig[i]);
    TPointMove2 &mi(move[i]);
    cv::Point2f pc= mi.Po + mi.DP;  // Current marker position.
    float sc= mi.So + mi.DS;  // Current size

    // Reset when number of tracking failures in a row exceeds a threshold
    if(mi.NTrackFailed>n_reset)
    {
      std::cerr<<"TrackKeyPoints2::MESSAGE:: blob tracking reset: "<<i<<std::endl;
      mi.Po= oi.pt;
      mi.So= oi.size;
      mi.DP= cv::Point2f(0.0,0.0);
      mi.DS= 0.0;
      mi.NTrackFailed= 0;
    }
    ++mi.NTrackFailed;

    // We will consider ROI around the current marker.
    if(0.25*s_width<sc)
      std::cerr<<"TrackKeyPoints2::WARNING:: s_width may be too small for a blob size: "<<s_width<<", "<<i<<", "<<sc<<std::endl;
    cv::Rect roi(pc.x-s_width*0.5, pc.y-s_width*0.5, s_width, s_width);
    if(roi.x<0)  {roi.width+= roi.x; roi.x= 0;}
    if(roi.y<0)  {roi.height+= roi.y; roi.y= 0;}
    if(roi.width<=0 || roi.height<=0)  continue;
    if(roi.x+roi.width>img_th.cols)  {roi.width= img_th.cols-roi.x;}
    if(roi.y+roi.height>img_th.rows)  {roi.height= img_th.rows-roi.y;}
    if(roi.width<=0 || roi.height<=0)  continue;

    // Count the nonzero pixels in ROI.
    if(is_thresholded)
    {
      float nonzero_ratio= cv::countNonZero(img_th(roi)) / (4*oi.size*oi.size);
      if(nonzero_ratio<nonzero_min || nonzero_ratio>nonzero_max)  continue;
    }

    // Detect a marker; if the number of keypoints is not 1, considered as an error.
    keypoints.clear();
    detector.detect(img_th(roi), keypoints);
    if(keypoints.size() == 0)
    {
      // std::cerr<<"TrackKeyPoints2::MESSAGE:: blob tracking failed: "<<i<<", # of keypoints: "<<keypoints.size()<<std::endl;
      continue;
    }
    else if(keypoints.size() > 1)
    {
      float vp_norm_min(0.0),j_min(-1);
      for(int j(0),j_end(keypoints.size()); j!=j_end; ++j)
      {
        cv::Point2f vp= cv::Point2f(roi.x,roi.y) + keypoints[j].pt - pc;
        float vp_norm(cv::sqrt(vp.x*vp.x + vp.y*vp.y));
        if(j_min<0 || vp_norm<vp_norm_min)
        {
          j_min= j;
          vp_norm_min= vp_norm;
        }
      }
      if(j_min!=0)  keypoints[0]= keypoints[j_min];
    }
    // Conversion to absolute position:
    keypoints[0].pt+= cv::Point2f(roi.x,roi.y);

    cv::Point2f vp= keypoints[0].pt - pc;
    float vs= std::fabs(keypoints[0].size - sc);
    float vp_norm(cv::sqrt(vp.x*vp.x + vp.y*vp.y));
    if(vp_norm<vp_max && vs<vs_max)
    {
      mi.DP= keypoints[0].pt - mi.Po;
      mi.DS= std::max(0.0f, keypoints[0].size - mi.So);
      mi.NTrackFailed= 0;
    }
    else
    {
      // std::cerr<<"TrackKeyPoints2::MESSAGE:: blob tracking failed: "<<i<<", vp,vs: "<<vp_norm<<", "<<vs<<std::endl;
    }
  }
}
//-------------------------------------------------------------------------------------------

TBlobTracker2Params::TBlobTracker2Params()
{
  SBDParams.filterByColor= 0;
  SBDParams.blobColor= 0;
  // Change thresholds
  SBDParams.minThreshold = 5;
  SBDParams.maxThreshold = 200;
  // Filter by Area.
  SBDParams.filterByArea = true;
  SBDParams.minArea= 10;
  SBDParams.maxArea= 150;
  // Filter by Circularity
  SBDParams.filterByCircularity = true;
  SBDParams.minCircularity = 0.10;
  SBDParams.maxCircularity = std::numeric_limits<float>::max();
  // Filter by Convexity
  SBDParams.filterByConvexity = true;
  SBDParams.minConvexity = 0.87;
  SBDParams.maxConvexity = std::numeric_limits<float>::max();
  // Filter by Inertia
  SBDParams.filterByInertia = true;
  SBDParams.minInertiaRatio = 0.01;
  SBDParams.maxInertiaRatio = std::numeric_limits<float>::max();

  // For preprocessing:
  ThresholdingImg= true;  // Thresholding the input image where Thresh*, NDilate1, NErode1 are used.
  ThreshH= 180;
  ThreshS= 255;
  ThreshV= 40;
  NDilate1= 2;
  NErode1= 2;
  // For keypoint tracking;
  SWidth= 30;  // Width of search ROI of each keypoint
  NonZeroMin= 0.5; // Minimum ratio of nonzero pixels in ROI over original keypoint area.
  NonZeroMax= 1.5; // Maximum ratio of nonzero pixels in ROI over original keypoint area.
  VPMax= 5.0;  // Maximum position change (too large one might be noise)
  VSMax= 1.0;  // Maximum size change (too large one might be noise)
  NReset= 3;  // When number of tracking failure in a row exceeds this, tracking is reset
  // For calibration:
  DistMaxCalib= 0.8;
  DSMaxCalib= 0.5;

  // For visualization:
  DSEmp= 4.0;
  DPEmp= 10.0;

  // For calibration:
  NCalibPoints= 10;
}
//-------------------------------------------------------------------------------------------

void WriteToYAML(const std::vector<TBlobTracker2Params> &blob_params, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs<<"BlobTracker2"<<"[";
  for(std::vector<TBlobTracker2Params>::const_iterator itr(blob_params.begin()),itr_end(blob_params.end()); itr!=itr_end; ++itr)
  {
    fs<<"{";
    #define PROC_VAR(x,y)  fs<<#x"_"#y<<itr->x.y;
    PROC_VAR(SBDParams,filterByColor       );
    PROC_VAR(SBDParams,blobColor           );
    PROC_VAR(SBDParams,minThreshold        );
    PROC_VAR(SBDParams,maxThreshold        );
    PROC_VAR(SBDParams,filterByArea        );
    PROC_VAR(SBDParams,minArea             );
    PROC_VAR(SBDParams,maxArea             );
    PROC_VAR(SBDParams,filterByCircularity );
    PROC_VAR(SBDParams,minCircularity      );
    PROC_VAR(SBDParams,maxCircularity      );
    PROC_VAR(SBDParams,filterByConvexity   );
    PROC_VAR(SBDParams,minConvexity        );
    PROC_VAR(SBDParams,maxConvexity        );
    PROC_VAR(SBDParams,filterByInertia     );
    PROC_VAR(SBDParams,minInertiaRatio     );
    PROC_VAR(SBDParams,maxInertiaRatio     );
    #undef PROC_VAR
    #define PROC_VAR(x)  fs<<#x<<itr->x;
    PROC_VAR(ThresholdingImg      );
    PROC_VAR(ThreshH      );
    PROC_VAR(ThreshS      );
    PROC_VAR(ThreshV      );
    PROC_VAR(NDilate1     );
    PROC_VAR(NErode1      );
    PROC_VAR(SWidth       );
    PROC_VAR(NonZeroMin   );
    PROC_VAR(NonZeroMax   );
    PROC_VAR(VPMax        );
    PROC_VAR(VSMax        );
    PROC_VAR(NReset       );
    PROC_VAR(DistMaxCalib );
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

void ReadFromYAML(std::vector<TBlobTracker2Params> &blob_params, const std::string &file_name)
{
  blob_params.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["BlobTracker2"];
  for(cv::FileNodeIterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
  {
    TBlobTracker2Params cf;
    #define PROC_VAR(x,y)  if(!(*itr)[#x"_"#y].empty())  (*itr)[#x"_"#y]>>cf.x.y;
    PROC_VAR(SBDParams,filterByColor       );
    PROC_VAR(SBDParams,blobColor           );
    PROC_VAR(SBDParams,minThreshold        );
    PROC_VAR(SBDParams,maxThreshold        );
    PROC_VAR(SBDParams,filterByArea        );
    PROC_VAR(SBDParams,minArea             );
    PROC_VAR(SBDParams,maxArea             );
    PROC_VAR(SBDParams,filterByCircularity );
    PROC_VAR(SBDParams,minCircularity      );
    PROC_VAR(SBDParams,maxCircularity      );
    PROC_VAR(SBDParams,filterByConvexity   );
    PROC_VAR(SBDParams,minConvexity        );
    PROC_VAR(SBDParams,maxConvexity        );
    PROC_VAR(SBDParams,filterByInertia     );
    PROC_VAR(SBDParams,minInertiaRatio     );
    PROC_VAR(SBDParams,maxInertiaRatio     );
    #undef PROC_VAR
    #define PROC_VAR(x)  if(!(*itr)[#x].empty())  (*itr)[#x]>>cf.x;
    PROC_VAR(ThresholdingImg      );
    PROC_VAR(ThreshH      );
    PROC_VAR(ThreshS      );
    PROC_VAR(ThreshV      );
    PROC_VAR(NDilate1     );
    PROC_VAR(NErode1      );
    PROC_VAR(SWidth       );
    PROC_VAR(NonZeroMin   );
    PROC_VAR(NonZeroMax   );
    PROC_VAR(VPMax        );
    PROC_VAR(VSMax        );
    PROC_VAR(NReset       );
    PROC_VAR(DistMaxCalib );
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
// class TBlobTracker2
//-------------------------------------------------------------------------------------------

void TBlobTracker2::Init()
{
  detector_= new cv::SimpleBlobDetector(params_.SBDParams);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::Preprocess(const cv::Mat &img, cv::Mat &img_th)
{
  if(params_.ThresholdingImg)
  {
    cv::cvtColor(img, img_th, cv::COLOR_BGR2HSV);
    cv::inRange(img_th, cv::Scalar(0, 0, 0), cv::Scalar(params_.ThreshH, params_.ThreshS, params_.ThreshV), img_th);
    cv::dilate(img_th,img_th,cv::Mat(),cv::Point(-1,-1), params_.NDilate1);
    cv::erode(img_th,img_th,cv::Mat(),cv::Point(-1,-1), params_.NErode1);
  }
  else
  {
    img.copyTo(img_th);
  }
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::Step(const cv::Mat &img)
{
  // Preprocess the image.
  Preprocess(img, img_th_);

  TrackKeyPoints2(
      img_th_, params_.ThresholdingImg, *detector_, keypoints_orig_, keypoints_move_,
      params_.SWidth, params_.NonZeroMin, params_.NonZeroMax, params_.VPMax, params_.VSMax, params_.NReset);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::Draw(cv::Mat &img)
{
  DrawPointMoves2(img, keypoints_move_, cv::Scalar(255,0,0), cv::Scalar(0,0,255),
      params_.DSEmp, params_.DPEmp, img_th_, params_.ThresholdingImg,
      params_.SWidth);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::Calibrate(const std::vector<cv::Mat> &images)
{
  std::cerr<<"Calibrating..."<<std::endl;
  Preprocess(images[0], img_th_);
  detector_->detect(img_th_, keypoints_orig_);
  std::cerr<<"keypoints_orig_.size()= "<<keypoints_orig_.size()<<std::endl;

  std::vector<TPointMove2> move;
  for(int i(1),i_end(images.size()); i<i_end; ++i)
  {
    Preprocess(images[i], img_th_);
    InitKeyPointMove(keypoints_orig_, move);
    TrackKeyPoints2(
        img_th_, params_.ThresholdingImg, *detector_, keypoints_orig_, move,
        params_.SWidth, params_.NonZeroMin, params_.NonZeroMax, params_.VPMax, params_.VSMax, params_.NReset);
    for(int j(move.size()-1); j>=0; --j)
    {
      float dp_norm(cv::sqrt(move[j].DP.x*move[j].DP.x + move[j].DP.y*move[j].DP.y));
      if(/*!move[j].mi.NTrackFailed>0 ||*/ move[j].DS>params_.DSMaxCalib || dp_norm>params_.DistMaxCalib)
        keypoints_orig_.erase(keypoints_orig_.begin()+j);
    }
  }
  std::cerr<<"keypoints_orig_.size()= "<<keypoints_orig_.size()<<std::endl;
  InitKeyPointMove(keypoints_orig_, keypoints_move_);
  std::cerr<<"Done."<<std::endl;
}
//-------------------------------------------------------------------------------------------

// Remove a keypoint around (x,y).
void TBlobTracker2::RemovePointAt(const cv::Point2f &p, const float &max_dist)
{
  int i_closest(-1);
  float d, d_closest(0.0f);
  for(int i(0),i_end(keypoints_orig_.size()); i!=i_end; ++i)
  {
    d= cv::norm(keypoints_orig_[i].pt-p);
    if(d<max_dist && (i_closest<0 || d<d_closest))
    {
      i_closest= i;
      d_closest= d;
    }
  }
  if(i_closest>=0)
  {
    keypoints_orig_.erase(keypoints_orig_.begin()+i_closest);
    keypoints_move_.erase(keypoints_move_.begin()+i_closest);
    InitKeyPointMove(keypoints_orig_, keypoints_move_);
  }
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::SaveCalib(const std::string &file_name) const
{
  WriteToYAML(keypoints_orig_, file_name);
}
//-------------------------------------------------------------------------------------------

void TBlobTracker2::LoadCalib(const std::string &file_name)
{
  ReadFromYAML(keypoints_orig_, file_name);
  InitKeyPointMove(keypoints_orig_, keypoints_move_);
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------

using namespace trick;
using namespace loco_rabbits;


TBlobTracker2 *PTracker(NULL);
template<typename T>
void OnTrack2(const TExtendedTrackbarInfo<T> &info, void*)
{
  TrackbarPrintOnTrack(info, NULL);
  if(PTracker!=NULL)  PTracker->Init();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  TBlobTracker2 tracker;
  tracker.Init();
  PTracker= &tracker;

  bool calib_request(true);

  std::string win("camera");
  cv::namedWindow(win,1);
  int trackbar_mode(0);

  TEasyVideoOut vout;
  vout.SetfilePrefix("/tmp/blobtr");

  int show_fps(0);
  cv::Mat frame, warped;
  for(int f(0);;++f)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    // Rotate90N(frame,frame,n_rotate90);

    tracker.Step(frame);
    tracker.Draw(frame);

    vout.Step(frame);
    vout.VizRec(frame);
    cv::imshow(win, frame);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    else if(char(c)=='W')  vout.Switch();
    else if(c=='p')  tracker.SaveCalib("/dev/stdout");
    else if(c=='C')
    {
      ++trackbar_mode;
      if(trackbar_mode==1)
      {
        CreateTrackbar<int>("ThreshV", win, &tracker.Params().ThreshV, 0, 255, 1, &TrackbarPrintOnTrack);
        CreateTrackbar<int>("NDilate1:", win, &tracker.Params().NDilate1, 0, 10, 1, &TrackbarPrintOnTrack);
        CreateTrackbar<int>("NErode1:", win, &tracker.Params().NErode1, 0, 10, 1, &TrackbarPrintOnTrack);
        CreateTrackbar<float>("SWidth:", win, &tracker.Params().SWidth, 0.0, 100.0, 0.1, &TrackbarPrintOnTrack);
        CreateTrackbar<float>("NonZeroMin:", win, &tracker.Params().NonZeroMin, 0.0, 20.0, 0.1, &TrackbarPrintOnTrack);
        CreateTrackbar<float>("NonZeroMax:", win, &tracker.Params().NonZeroMax, 0.0, 20.0, 0.1, &TrackbarPrintOnTrack);
        CreateTrackbar<float>("VPMax:", win, &tracker.Params().VPMax, 0.0, 20.0, 0.1, &TrackbarPrintOnTrack);
        CreateTrackbar<float>("VSMax:", win, &tracker.Params().VSMax, 0.0, 20.0, 0.1, &TrackbarPrintOnTrack);
        CreateTrackbar<int>("NReset:", win, &tracker.Params().NReset, 0, 20, 1, &TrackbarPrintOnTrack);
      }
      else if(trackbar_mode==2)
      {
        // Remove trackbars from window.
        cv::destroyWindow(win);
        cv::namedWindow(win,1);
        CreateTrackbar<float>("SBDParams.minArea", win, &tracker.Params().SBDParams.minArea, 0.0, 20000.0, 1.0, &OnTrack2);
        CreateTrackbar<float>("SBDParams.maxArea", win, &tracker.Params().SBDParams.maxArea, 0.0, 20000.0, 1.0, &OnTrack2);
        CreateTrackbar<float>("SBDParams.minCircularity:", win, &tracker.Params().SBDParams.minCircularity, 0.0, 10.0, 0.01, &OnTrack2);
        CreateTrackbar<float>("SBDParams.minConvexity:", win, &tracker.Params().SBDParams.minConvexity, 0.0, 10.0, 0.01, &OnTrack2);
        CreateTrackbar<float>("SBDParams.minInertiaRatio:", win, &tracker.Params().SBDParams.minInertiaRatio, 0.0, 10.0, 0.01, &OnTrack2);
      }
      else
      {
        trackbar_mode= 0;
        // Remove trackbars from window.
        cv::destroyWindow(win);
        cv::namedWindow(win,1);
      }
    }
    else if(c=='c' || calib_request)
    {
      std::vector<cv::Mat> frames;
      for(int i(0); i<tracker.Params().NCalibPoints; ++i)
      {
        cap >> frame;
        // Rotate90N(frame,frame,n_rotate90);
        frames.push_back(frame.clone());
      }
      tracker.Calibrate(frames);
      calib_request= false;
    }
    // usleep(10000);
    if(show_fps==0)
    {
      std::cerr<<"FPS: "<<vout.FPS()<<std::endl;
      show_fps= vout.FPS()*4;
    }
    --show_fps;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------

