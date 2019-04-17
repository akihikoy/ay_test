/* Compile:
x++ color_detector.cpp -- -lopencv_core -lopencv_imgproc -lopencv_highgui
*/

// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>  // cvtColor
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <cstdio>
#include <cassert>


// Original: http://imagingsolution.blog107.fc2.com/blog-entry-248.html
//---------------------------------------------------------------
//【関数名　】：cv_ColorExtraction
//【処理概要】：色抽出
//【引数　　】：src_img        = 入力画像(8bit3ch)
//　　　　　　：dst_img        = 出力画像(8bit3ch)
//　　　　　　：code        = 色空間の指定（CV_BGR2HSV,CV_BGR2Labなど）
//　　　　　　：ch1_lower    = ch1のしきい値(小)
//　　　　　　：ch1_upper    = ch1のしきい値(大)
//　　　　　　：ch2_lower    = ch2のしきい値(小)
//　　　　　　：ch2_upper    = ch2のしきい値(大)
//　　　　　　：ch3_lower    = ch3のしきい値(小)
//　　　　　　：ch3_upper    = ch3のしきい値(大)
//【戻り値　】：なし
//【備考　　】：lower <= upperの場合、lower以上upper以下の範囲を抽出、
//　　　　　　：lower >  upperの場合、upper以下lower以上の範囲を抽出します。
//---------------------------------------------------------------
void cv_ColorExtraction(const cv::Mat &src_img, cv::Mat &dst_img,
  int code, const cv::Vec3b &lower, const cv::Vec3b &upper)
{
  int i, k;

  assert(src_img.type()==CV_8UC3);

  cv::Mat lut;

  //codeに基づいたカラー変換
  cv::Mat Color_img;
  cv::cvtColor(src_img, Color_img, code);

  //3ChのLUT作成
  lut.create(256, 1, CV_8UC3);

  for (i = 0; i < 256; i++)
  {
    cv::Vec3b val;
    for (k = 0; k < 3; k++)
    {
      if (lower[k] <= upper[k])
      {
        if ((lower[k] <= i) && (i <= upper[k]))
          val[k] = 255;
        else
          val[k] = 0;
      }
      else
      {
        if ((i <= upper[k]) || (lower[k] <= i))
          val[k] = 255;
        else
          val[k] = 0;
      }
    }
    //LUTの設定
    lut.at<cv::Vec3b>(i,0)= val;
  }

  //3ChごとのLUT変換（各チャンネルごとに２値化処理）
  cv::LUT(Color_img, lut, Color_img);

  cv::Mat ch_imgs[3];
  cv::split(Color_img, ch_imgs);

  //3Ch全てのANDを取り、マスク画像を作成する。
  cv::Mat Mask_img;
  cv::bitwise_and(ch_imgs[0], ch_imgs[1], Mask_img);
  cv::bitwise_and(Mask_img, ch_imgs[2], Mask_img);
  cv::imshow("debug", Mask_img);

  //入力画像(src_img)のマスク領域を出力画像(dst_img)へコピーする
  dst_img.create(src_img.rows, src_img.cols, CV_8UC3);
  dst_img= cv::Scalar(0,0,0);
  src_img.copyTo(dst_img, Mask_img);
}

int main(int argc, char **argv)
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

  cv::namedWindow("camera",1);
  cv::namedWindow("detected",1);
  cv::namedWindow("debug",1);
  cv::Mat frame, detected;
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);
    // Human skin:
    cv_ColorExtraction(frame,detected,CV_BGR2HSV, cv::Vec3b(0,80,0), cv::Vec3b(10,255,255));
    // cv_ColorExtraction(frame,detected,CV_BGR2HSV, cv::Vec3b(177,80,102), cv::Vec3b(22,131,200));
    // Nexus' red:
    // cv_ColorExtraction(frame,detected,CV_BGR2HSV, cv::Vec3b(161,80,83), cv::Vec3b(25,188,206));
    cv::imshow("detected", detected);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
