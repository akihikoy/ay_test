#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <iomanip>

// g++ -I -Wall flow_finder_2.cpp -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_legacy -lopencv_imgproc -lopencv_highgui

/*!\brief calculate optical flow */
class TOpticalFlow
{
public:

  TOpticalFlow();
  ~TOpticalFlow()  {}

  void CalcLK(const cv::Mat &prev, const cv::Mat &curr);

  void GetAngleDistImg(cv::Mat &img_dist, cv::Mat &img_angle);

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

void TOpticalFlow::CalcLK(const cv::Mat &prev, const cv::Mat &curr)
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

void TOpticalFlow::GetAngleDistImg(cv::Mat &img_dist, cv::Mat &img_angle)
{
  img_dist.create(rows_, cols_, CV_32FC1);
  img_angle.create(rows_, cols_, CV_32FC1);

  typedef cv::MatIterator_<float> t_itr;
  t_itr ivx(velx_.begin<float>()), ivy(vely_.begin<float>());
  t_itr itr_d(img_dist.begin<float>()), itr_a(img_angle.begin<float>());
  t_itr itr_d_last(img_dist.end<float>());

  for(; itr_d!=itr_d_last; ++ivx,++ivy,++itr_d,++itr_a)
  {
    *itr_d= std::sqrt((*ivx)*(*ivx)+(*ivy)*(*ivy));
    *itr_a= std::atan2(*ivy,*ivx);
    // std::cerr<<*itr_d<<", "<<*itr_a<<std::endl;
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
  cv::namedWindow("img_dist",1);
  cv::namedWindow("img_angle",1);
  cv::Mat frame, frame_old, frame_gray, mask, mask2, img_dist, img_angle;
  cap >> frame_old;
  cv::cvtColor(frame_old,frame_old,CV_BGR2GRAY);
  for(int f(0);;++f)
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

    // optflow.CalcLK(frame_old, frame);
    optflow.CalcLK(frame, frame_old);  // NOTE: Inverting the previous and current
    frame_old= frame.clone();

    // cv::cvtColor(frame,frame,CV_GRAY2RGB);

    optflow.GetAngleDistImg(img_dist, img_angle);

    /* 0: Binary
      1: Binary Inverted
      2: Threshold Truncated
      3: Threshold to Zero
      4: Threshold to Zero Inverted
    */
    cv::threshold(img_dist, mask, /*thresh=*/5.0, /*maxval=*/1.0, cv::THRESH_BINARY);
    mask.convertTo(mask,CV_8UC1);
    cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
    cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);

    /*apply mask*/{
      cv::Mat tmp(img_angle.size(),img_angle.type(),-M_PI);
      img_angle.copyTo(tmp, mask);
      img_angle= tmp;
    }
    /*apply mask*/{
      cv::Mat tmp(img_dist.size(),img_dist.type(),0.0);
      img_dist.copyTo(tmp, mask);
      img_dist= tmp;
    }

    // Countour
    std::vector<std::vector<cv::Point> > contours;
    mask2= mask.clone();
    cv::findContours(mask2,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    for(int i(0); i<contours.size(); ++i)
    {
      // double area= cv::contourArea(contours[i]);
      // // std::cerr<<"area= "<<area<<std::endl;
      // // Remove small and big area:
      // if(area<10 || 0.05*(double(mask.rows*mask.cols))<area)
      // {
        // const cv::Point *pts= (const cv::Point*) cv::Mat(contours[i]).data;
        // int npts= cv::Mat(contours[i]).rows;
        // cv::fillPoly(mask, &pts, &npts, /*ncontours=*/1, cv::Scalar(0), /*lineType=*/8);

        // contours[i].clear();
      // }
      // else
      // {
        // // cv::drawContours(mask, contours, i, CV_RGB(255,0,0), /*thickness=*/1, /*linetype=*/8);
        // cv::drawContours(img_dist, contours, i, cv::Scalar(1.0), /*thickness=*/1, /*linetype=*/8);
        // cv::drawContours(img_angle, contours, i, cv::Scalar(M_PI), /*thickness=*/1, /*linetype=*/8);
      // }

      // cv::RotatedRect bound= cv::minAreaRect(contours[i]);
      // if(bound.size.height/bound.size.width>3.0)
      // {
        // /*Draw bount box*/{
          // cv::Point2f pts[4];
          // cv::Point ptsi[4];
          // bound.points(pts);
          // int n_pts(4);
          // for(int p(0);p<4;++p)  ptsi[p]= pts[p];
          // const cv::Point *ppts[4]= {&ptsi[0],&ptsi[1],&ptsi[2],&ptsi[3]};
          // cv::polylines(img_dist, ppts, &n_pts, /*ncontours=*/1, /*isClosed=*/true, cv::Scalar(1.0), /*thickness=*/1, /*lineType=*/8);
          // cv::polylines(img_angle, ppts, &n_pts, /*ncontours=*/1, /*isClosed=*/true, cv::Scalar(M_PI), /*thickness=*/1, /*lineType=*/8);
        // }
      // }

      double avr_angle(0.0), avr_dist(0.0);
      // Compute average angle and distance from optical flow image:
      // cv::Rect bound= cv::boundingRect(contours[i]);
      // int num_points(0);
      // for (int px(bound.x); px<bound.x+bound.width; ++px)
      // {
        // for (int py(bound.y); py<bound.y+bound.height; ++py)
        // {
          // // if(cv::pointPolygonTest(contours[i],cv::Point2f(px,py),/*measureDist=*/false)>=0.0)
          // {
            // avr_dist+= img_dist.at<float>(py,px);
            // avr_angle+= img_angle.at<float>(py,px);
            // ++num_points;
          // }
        // }
      // }
      // avr_dist/= double(num_points);
      // avr_angle/= double(num_points);
      // Compute average angle and distance from bounding box:
      cv::RotatedRect bound= cv::minAreaRect(contours[i]);
      {
        cv::Point2f pts[4];
        bound.points(pts);
        cv::Vec2f v,v1(pts[1]-pts[0]),v2(pts[2]-pts[1]);
        if(cv::norm(v1)>cv::norm(v2))  v= v1;
        else  v= v2;

        avr_dist= cv::norm(v);
        avr_angle= std::atan2(v[1],v[0]);
      }
      if(bound.size.height/bound.size.width>3.0)
      {
        /*Draw bount box*/{
          cv::Point2f pts[4];
          cv::Point ptsi[4];
          bound.points(pts);
          int n_pts(4);
          for(int p(0);p<4;++p)  ptsi[p]= pts[p];
          const cv::Point *ppts[4]= {&ptsi[0],&ptsi[1],&ptsi[2],&ptsi[3]};
          cv::polylines(img_dist, ppts, &n_pts, /*ncontours=*/1, /*isClosed=*/true, cv::Scalar(1.0), /*thickness=*/1, /*lineType=*/8);
          cv::polylines(img_angle, ppts, &n_pts, /*ncontours=*/1, /*isClosed=*/true, cv::Scalar(M_PI), /*thickness=*/1, /*lineType=*/8);
        }
      }

      // Moments and center:
      cv::Moments mu= cv::moments(contours[i]);
      cv::Point2d center(mu.m10/mu.m00, mu.m01/mu.m00);

      // if( (bound.height>15 && double(bound.height)/double(bound.width)>2.0)
        // || (std::fabs(avr_vy)>1.0) )
      {
        std::cerr<<center<<", "<<cv::Point2d(avr_dist,avr_angle)<<", "<<bound.angle<<std::endl;

        // cv::rectangle(img_dist, bound, cv::Scalar(1.0));
        // cv::rectangle(img_angle, bound, cv::Scalar(M_PI));

        float len(1.0);
        cv::line(img_angle, center, center+cv::Point2d(len*avr_dist*std::cos(avr_angle),len*avr_dist*std::sin(avr_angle)),
                cv::Scalar(M_PI), /*thickness=*/3, /*line_type=*/CV_AA, 0);
      }
    }


    // std::stringstream file_name;
    // file_name<<"frame/frame"<<std::setfill('0')<<std::setw(4)<<f<<".jpg";
    // cv::imwrite(file_name.str(), (img_angle+M_PI)/(2.0*M_PI)*255.0);
    // std::cout<<"Saved "<<file_name.str()<<std::endl;
    // if(f==10000)  f=0;

    cv::imshow("camera", frame);
    cv::imshow("img_dist", img_dist);
    cv::imshow("img_angle", (img_angle+M_PI)/(2.0*M_PI));
    int c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}

