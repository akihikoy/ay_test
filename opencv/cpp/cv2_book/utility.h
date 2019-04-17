/*! \file    utility.h
    \brief   �桼�ƥ���ƥ��ؿ���(�إå�) */
//------------------------------------------------------------------------------
#ifndef utility_h
#define utility_h
//------------------------------------------------------------------------------
#include <opencv/cv.h>
#include <cstdlib>
//------------------------------------------------------------------------------

const cv::Scalar& NumberedColor(unsigned int i);

template <typename T>
inline T Square(const T &x)
{
  return x*x;
}

// [0,1]��������֤�
inline double Rand()
{
  return static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
}

// [-r,r]x[-r,r]�Υ�����������֤� (��ɸ�˥Υ�����ä��뤿��˻Ȥ�)
inline cv::Point PtNoise(int r=3)
{
  return cv::Point(2.0*r*Rand()-r,2.0*r*Rand()-r);
}

// img ���濴(meanx,meany),��ʬ������ cov �Υ�����������ʱ�(��Ψ̩�٤����������)������
// xscale,yscale �ϥ��������, offset �ϥ��ե��å�, color �ʲ��϶���������
// N ���ʱߤ��������
void DrawGaussianEllipse(cv::Mat &img, const cv::Mat &cov,
    const double &meanx, const double &meany,
    const double &xscale, const double &yscale, const cv::Point &offset,
    const cv::Scalar &color, int thickness=1, int line_type=8, int shift=0,
    int N=25);

// �� N �ѷ��� img ������. pos ���濴����,radius ��Ⱦ��, color �ʲ��϶���������
void DrawRegularPolygon(int N, cv::Mat &img, const cv::Point &pos, int radius,
    const cv::Scalar &color, int thickness=1, int line_type=8, int shift=0);

#endif // utility_h
