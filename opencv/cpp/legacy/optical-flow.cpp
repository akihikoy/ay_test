#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>
#include "cap_open.h"

// g++ -I -Wall optical-flow.cpp -o optical-flow.out -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_legacy -lopencv_imgproc -lopencv_highgui -lopencv_videoio

/*!\brief calculate optical flow */
class TOpticalFlow
{
public:

  TOpticalFlow();
  ~TOpticalFlow()  {}

  void CalcBM(const cv::Mat &prev, const cv::Mat &curr);

  void DrawOnImg(cv::Mat &img, const cv::Scalar &color, int thickness=1,
    int line_type=8, int shift=0);

  cv::Point GetPosOnImg(int i, int j) const
      {return GetPosOnImg(cv::Point(i,j));}
  cv::Point GetPosOnImg(const cv::Point &p) const
      {return cv::Point(p.x*shift_size_.width+block_size_.width/2,
                        p.y*shift_size_.height+block_size_.height/2);}

  const cv::Size& BlockSize() const {return block_size_;}
  const cv::Size& ShiftSize() const {return shift_size_;}
  const cv::Size& MaxRange() const {return max_range_;}

  void SetBlockSize(int bs)  {block_size_= cv::Size(bs,bs);}
  void SetBlockSize(const cv::Size &bs)  {block_size_= bs;}
  void SetShiftSize(int ss)  {shift_size_= cv::Size(ss,ss);}
  void SetShiftSize(const cv::Size &ss)  {shift_size_= ss;}
  void SetMaxRange(int mr)  {max_range_= cv::Size(mr,mr);}
  void SetMaxRange(const cv::Size &mr)  {max_range_= mr;}

  bool UsePrevious() const {return use_previous_;}
  void SetUsePrevious(bool up)  {use_previous_= up;}

  const cv::Mat& VelX() const {return velx_;}
  const cv::Mat& VelY() const {return vely_;}

  const float& VelXAt(int i, int j) const {return velx_.at<float>(j, i);}
  const float& VelYAt(int i, int j) const {return vely_.at<float>(j, i);}

  int  Rows() const {return rows_;}
  int  Cols() const {return cols_;}

private:

  TOpticalFlow(const TOpticalFlow&);
  const TOpticalFlow& operator=(const TOpticalFlow&);

  bool first_exec_;

  cv::Size block_size_, shift_size_, max_range_;
  bool use_previous_;

  cv::Mat velx_, vely_, curr_;

  int rows_, cols_;
};
//------------------------------------------------------------------------------

//==============================================================================
// class TOpticalFlow
//==============================================================================

TOpticalFlow::TOpticalFlow()
  :
    first_exec_   (true),
    block_size_   (cv::Size(8,8)),
    shift_size_   (cv::Size(20,20)),
    max_range_    (cv::Size(25,25)),
    use_previous_ (true),
    rows_         (0),
    cols_         (0)
{
}
//------------------------------------------------------------------------------

void TOpticalFlow::CalcBM(const cv::Mat &prev, const cv::Mat &curr)
{
  // WARNING: +1 is highly experimental, depends on version, remove it if seeing an error, OpenCV Error: Sizes of input arguments do not match () in cvCalcOpticalFlowBM
  rows_= floor(static_cast<double>(curr.rows-block_size_.height)
                / static_cast<double>(shift_size_.height))+1;
  cols_= floor(static_cast<double>(curr.cols-block_size_.width)
                / static_cast<double>(shift_size_.width))+1;

  velx_.create(rows_, cols_, CV_32FC1);
  vely_.create(rows_, cols_, CV_32FC1);

  int use_prev= ((!first_exec_ && use_previous_)? 1 : 0);
  if (use_prev)
  {
    velx_*=0.5;
    vely_*=0.5;
  }

  curr_= curr.clone();
  CvMat prev2(prev), curr2(curr), velx2(velx_), vely2(vely_);
  cvCalcOpticalFlowBM (&prev2, &curr2, block_size_, shift_size_, max_range_,
    use_prev, &velx2, &vely2);
  first_exec_= false;
}
//------------------------------------------------------------------------------

void TOpticalFlow::DrawOnImg(cv::Mat &img, const cv::Scalar &color,
  int thickness, int line_type, int shift)
{
  int dx,dy;
  for (int i(0); i<cols_; i+=1)
  {
    for (int j(0); j<rows_; j+=1)
    {
      cv::Scalar mean, sd;
      cv::Mat block(curr_(cv::Rect(GetPosOnImg(i,j)-cv::Point(block_size_.width/2,block_size_.height/2), block_size_)));
      cv::meanStdDev(block, mean, sd);
      if(sd[0]<5.0)  continue;
      std::cerr<<mean<<", "<<sd<<std::endl;

      dx = cvRound(VelXAt(i,j));
      dy = cvRound(VelYAt(i,j));
      cv::line (img, GetPosOnImg(i,j), GetPosOnImg(i,j)+cv::Point(dx,dy),
              color, thickness, line_type, shift);
    }
  }
}
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  TOpticalFlow optflow;

  // #define USING_BKG_SBTR

  #ifdef USING_BKG_SBTR
  cv::Ptr<cv::BackgroundSubtractorMOG2> bkg_sbtr= cv::createBackgroundSubtractorMOG2(/*int history=*/10, /*double varThreshold=*/5.0, /*bool detectShadows=*/true);
  #endif

  cv::namedWindow("camera",1);
  cv::Mat frame, frame_old, velx, vely;
  cap >> frame_old;
  cv::cvtColor(frame_old,frame_old,CV_BGR2GRAY);
  for(int i(0);;++i)
  {
    cap >> frame;
    #ifdef USING_BKG_SBTR
      bkg_sbtr->apply(frame,frame);
      cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), 1);
      cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), 3);
    #else
      cv::cvtColor(frame,frame,CV_BGR2GRAY);
    #endif

    // medianBlur(frame, frame, 9);

    // CvTermCriteria criteria= cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 64, 0.01);
    // cvCalcOpticalFlowHS(&frame_old, &frame, 0, &velx, &vely, 100.0, criteria);
    // CalcOpticalFlowBM(frame_old, frame, velx, vely);
    optflow.CalcBM(frame_old, frame);

    optflow.DrawOnImg(frame, CV_RGB(255,255,255), 1, CV_AA, 0);

    // std::stringstream file_name;
    // file_name<<"frame/frame"<<std::setfill('0')<<std::setw(4)<<i<<".jpg";
    // cv::imwrite(file_name.str(), frame);
    // std::cout<<"Saved "<<file_name.str()<<std::endl;
    // if(i==10000)  i=0;

    cv::imshow("camera", frame);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
    frame_old= frame;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
