/*! \file    utility.h
    \brief   ユーティリティ関数群(ヘッダ) */
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

// [0,1]の乱数を返す
inline double Rand()
{
  return static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
}

// [-r,r]x[-r,r]のランダムな点を返す (座標にノイズを加えるために使う)
inline cv::Point PtNoise(int r=3)
{
  return cv::Point(2.0*r*Rand()-r,2.0*r*Rand()-r);
}

// img に中心(meanx,meany),共分散行列 cov のガウシアンの楕円(確率密度が一定の点群)を描画
// xscale,yscale はスケーリング, offset はオフセット, color 以下は曲線の設定
// N は楕円の補間点数
void DrawGaussianEllipse(cv::Mat &img, const cv::Mat &cov,
    const double &meanx, const double &meany,
    const double &xscale, const double &yscale, const cv::Point &offset,
    const cv::Scalar &color, int thickness=1, int line_type=8, int shift=0,
    int N=25);

// 正 N 角形を img に描画. pos は中心位置,radius は半径, color 以下は曲線の設定
void DrawRegularPolygon(int N, cv::Mat &img, const cv::Point &pos, int radius,
    const cv::Scalar &color, int thickness=1, int line_type=8, int shift=0);

#endif // utility_h
