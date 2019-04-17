/*! \file    optflow.h
    \brief   オプティカルフロー(ヘッダ) */
//------------------------------------------------------------------------------
#ifndef optflow_h
#define optflow_h
//------------------------------------------------------------------------------
#include <opencv/cv.h>
//------------------------------------------------------------------------------

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

  cv::Mat velx_, vely_;

  int rows_, cols_;
};

#endif // optflow_h
