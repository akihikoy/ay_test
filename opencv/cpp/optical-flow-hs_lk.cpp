#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>

// g++ -I -Wall optical-flow-hs_lk.cpp -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_legacy -lopencv_imgproc -lopencv_highgui

/*!\brief calculate optical flow */
class TOpticalFlow
{
public:

  TOpticalFlow();
  ~TOpticalFlow()  {}

  void CalcHS(const cv::Mat &prev, const cv::Mat &curr);

  void DrawOnImg(cv::Mat &img, const cv::Scalar &color, int thickness=1,
    int line_type=8, int shift=0);

  cv::Point GetPosOnImg(int i, int j) const
      {return GetPosOnImg(cv::Point(i,j));}
  cv::Point GetPosOnImg(const cv::Point &p) const
      {return p;}

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

void TOpticalFlow::CalcHS(const cv::Mat &prev, const cv::Mat &curr)
{
  cols_= curr.cols;
  rows_= curr.rows;
  velx_.create(rows_, cols_, CV_32FC1);
  vely_.create(rows_, cols_, CV_32FC1);
  velx_= cv::Scalar(0);
  vely_= cv::Scalar(0);

  curr_= curr.clone();
  CvMat prev2(prev), curr2(curr), velx2(velx_), vely2(vely_);

  // Using HS:
  // CvTermCriteria criteria= cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 64, 0.01);
  // cvCalcOpticalFlowHS(&prev2, &curr2, 0, &velx2, &vely2, 100.0, criteria);
  // Using LK:
  cvCalcOpticalFlowLK(&prev2, &curr2, cv::Size(3,3), &velx2, &vely2);
}
//------------------------------------------------------------------------------

void TOpticalFlow::DrawOnImg(cv::Mat &img, const cv::Scalar &color,
  int thickness, int line_type, int shift)
{
  cv::Scalar col(color);
  float vx,vy,dist,angle;
  int dx,dy;
  const float len(0.1);
  // const float len(0.5);
  // const float len(1.0);
  // const float len(2.0);
  int step(1);
  for (int i(step); i<cols_-step; i+=step)
  {
    for (int j(step); j<rows_-step; j+=step)
    {
      // cv::Scalar mean, sd;
      // cv::Mat block(curr_(cv::Rect(GetPosOnImg(i,j)-cv::Point(block_size_.width/2,block_size_.height/2), block_size_)));
      // cv::meanStdDev(block, mean, sd);
      // if(sd[0]<5.0)  continue;
      // std::cerr<<mean<<", "<<sd<<std::endl;

      vx= VelXAt(i,j);
      vy= VelYAt(i,j);
      dist= std::sqrt(vx*vx+vy*vy);
      if(dist<1.0 || 500.0<dist)  continue;
      angle= std::atan2(vy,vx);
      // if(0.4*M_PI<std::fabs(angle) && std::fabs(angle)<0.6*M_PI)  col= CV_RGB(255,0,0);
      // else col= color;
      col= CV_RGB(255.0*std::fabs(std::cos(angle)),255.0*std::fabs(std::sin(angle)),0.0);
      cv::line (img, GetPosOnImg(i,j), GetPosOnImg(i,j)+cv::Point(len*vx,len*vy),
              col, thickness, line_type, shift);
    }
  }
}
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  TOpticalFlow optflow;

  // #define USING_BKG_SBTR

  #ifdef USING_BKG_SBTR
  cv::BackgroundSubtractorMOG2 bkg_sbtr(/*int history=*/10, /*double varThreshold=*/5.0, /*bool detectShadows=*/true);
  #endif

  cv::namedWindow("camera",1);
  cv::Mat frame, frame_old, frame_gray, mask;
  cap >> frame_old;
  cv::cvtColor(frame_old,frame_old,CV_BGR2GRAY);
  for(int i(0);;++i)
  {
    cap >> frame;
    #ifdef USING_BKG_SBTR
      bkg_sbtr(frame,mask);
      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      // Use mask as an input image:
      // frame= mask;
      // Use gray scale of original image applied the mask:
      cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);
      frame= cv::Scalar(0.0,0.0,0.0);
      frame_gray.copyTo(frame, mask);
    #else
      cv::cvtColor(frame,frame,CV_BGR2GRAY);
    #endif

    // medianBlur(frame, frame, 9);

    optflow.CalcHS(frame_old, frame);
    frame_old= frame.clone();

    cv::cvtColor(frame,frame,CV_GRAY2RGB);
    optflow.DrawOnImg(frame, CV_RGB(255,255,255), 1, CV_AA, 0);

    // std::stringstream file_name;
    // file_name<<"frame/frame"<<std::setfill('0')<<std::setw(4)<<i<<".jpg";
    // cv::imwrite(file_name.str(), frame);
    // std::cout<<"Saved "<<file_name.str()<<std::endl;
    // if(i==10000)  i=0;

    cv::imshow("camera", frame);
    int c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}

