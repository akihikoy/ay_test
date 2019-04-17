/*! \file    optflow-gmmem.cpp
    \brief   オプティカルフローをGMM+EMアルゴリズムでクラスタリング */
//------------------------------------------------------------------------------
#include "seq-em.h"
#include "optflow.h"
#include "utility.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <iomanip>
#include <sstream>
//------------------------------------------------------------------------------
using namespace std;

int main(int argc, char** argv)
{
  cv::VideoCapture capture(0); // デフォルトカメラを開く
  if(!capture.isOpened())  // カメラが開けたかどうか
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }

  cv::namedWindow("Capture",1);
  cv::namedWindow("OpticalFlow",1);
  cv::Mat prev_img, curr_img;
  cv::Mat disp_img, optflow_img;

  TOpticalFlow  optflow;
  optflow.SetUsePrevious(true);
  optflow.SetBlockSize(10);
  optflow.SetShiftSize(20);
  optflow.SetMaxRange(12);

  // オプティカルフローのノルムの2乗がこの値以下ならデータから除外する
  int min_norm_sq(Square(4));

  TGMM2D gmm;
  gmm.SetParams().NModels= 4;

  bool gmm_fixed(false);
  bool first_loop(true);
  int write_idx(0);

  capture >> disp_img;
  cv::cvtColor(disp_img,curr_img,CV_BGR2GRAY);

  while (true)
  {
    try
    {
      swap(curr_img,prev_img);
      capture >> disp_img;
      cv::cvtColor(disp_img,curr_img,CV_BGR2GRAY);
      if (first_loop)  {first_loop=false; continue;}

      // オプティカルフローを計算:
      optflow.CalcBM(prev_img, curr_img);

      // オプティカルフローを描画(キャプチャ画像上)
      optflow.DrawOnImg(disp_img, CV_RGB(255,255,255), 1, CV_AA, 0);

      // オプティカルフローを描画(dx,dy の分布をxy平面にプロット)
      optflow_img.create(400,400, CV_8UC3);
      cv::rectangle (optflow_img, cv::Point(0,0),
                     cv::Point(optflow_img.cols-1,optflow_img.rows-1),
                     CV_RGB(255,255,255), CV_FILLED);
      double xscale(0.99*optflow_img.cols/optflow.MaxRange().width/2);
      double yscale(0.99*optflow_img.rows/optflow.MaxRange().height/2);

      // データ点の追加
      float dx,dy;
      for (int i(0),cols(optflow.Cols()); i<cols; ++i)
      {
        for (int j(0),rows(optflow.Rows()); j<rows; ++j)
        {
          dx = optflow.VelXAt(i,j);
          dy = optflow.VelYAt(i,j);
          if(dx*dx+dy*dy>=min_norm_sq)
            gmm.AddToData(dx,dy);
        }
      }

      // GMM が保管している学習データをオプティカルフロー空間にプロット
      for (int s(0),smax(gmm.NumOfSamples()); s<smax; ++s)
      {
        dx = optflow_img.cols/2 + xscale*gmm.GetSamples().at<float>(s,0);
        dy = optflow_img.rows/2 + yscale*gmm.GetSamples().at<float>(s,1);
        cv::circle(optflow_img, cv::Point(dx,dy)+PtNoise(), 1,
                    cv::Scalar(200,200,200), CV_FILLED);
      }

      if(gmm.EMExecutable())
      {
        if (!gmm_fixed)
          gmm.TrainByEM();

        // 現在のオプティカルフローをクラスタリングし, 描画

        cv::Mat probs;
        for (int i(0),cols(optflow.Cols()); i<cols; ++i)
        {
          for (int j(0),rows(optflow.Rows()); j<rows; ++j)
          {
            dx = optflow.VelXAt(i,j);
            dy = optflow.VelYAt(i,j);
            if(dx*dx+dy*dy<min_norm_sq)  continue;
            int m_idx= gmm.Predict(dx,dy,&probs); // オプティカルフローのクラス

            // カメラ画像上でオプティカルフローのクラスを多角形で可視化
            int radius= 0.5*probs.at<float>(0,m_idx)
                        *static_cast<float>(optflow.ShiftSize().width);
            DrawRegularPolygon(m_idx+3, disp_img, optflow.GetPosOnImg(i,j),
                                radius, NumberedColor(m_idx));

            // オプティカルフロー空間にプロット(クラスを多角形で可視化)
            dx = optflow_img.cols/2 + xscale*dx;
            dy = optflow_img.rows/2 + yscale*dy;
            DrawRegularPolygon(m_idx+3, optflow_img, cv::Point(dx,dy)+PtNoise(),
                                5, NumberedColor(m_idx), 1);
          }
        }

        // 各クラスのガウシアンを楕円で可視化
        const cv::Mat em_means(gmm.GetEMModel().get_means());
        const cv::Point em_offset(optflow_img.cols/2,optflow_img.rows/2);
        for(int i(0); i<gmm.Params().NModels; ++i)
        {
          const cv::Mat cov(gmm.GetEMModel().get_covs()[i]);
          // cout<<"mean["<<i<<"]="<<em_means.at<double>(i,0)<<", "
              // <<em_means.at<double>(i,1)<<endl;
          // cout<<"cov["<<i<<"]=("
              // <<cov.at<double>(0,0)<<", "<<cov.at<double>(0,1)<<"; "
              // <<cov.at<double>(1,0)<<", "<<cov.at<double>(1,1)<<")"<<endl;
          DrawGaussianEllipse(optflow_img, cov,
              em_means.at<double>(i,0), em_means.at<double>(i,1),
              xscale, yscale, em_offset, NumberedColor(i), 2);
        }
      }  // training GMM

      cv::imshow ("Capture", disp_img);
      cv::imshow ("OpticalFlow", optflow_img);
    }
    catch(const cv::Exception &e)
    {
      cerr<<"OpenCV exception has been caught:"<<endl;
      cerr<<e.err<<endl;
      first_loop= true;
    }

    int c(cv::waitKey(10));  // キー入力
    if(c=='\x1b'||c=='q'||c=='Q') break;  // Esc または q キーで終了
    else if(c=='w'||c=='W')  // w キー: スナップショットを保存
    {
      stringstream idx;
      idx<<setfill('0')<<setw(4)<<write_idx;
      vector<int> option;
      option.push_back(CV_IMWRITE_JPEG_QUALITY);
      option.push_back(100);
      cerr<<"write images to: {frame,optflow}"<<idx.str()<<".jpg"<<endl;
      cv::imwrite("frame"+idx.str()+".jpg", disp_img, option);
      cv::imwrite("optflow"+idx.str()+".jpg", optflow_img, option);
      ++write_idx;
    }
    else if(c=='p'||c=='P')  // p キー: オプティカルフローで前回の結果を使う/使わない
      optflow.SetUsePrevious(!optflow.UsePrevious());
    else if(c=='f'||c=='F')  // f キー: GMM を固定/固定解除
      gmm_fixed= !gmm_fixed;
    else if(c=='c'||c=='C')  // c キー: GMM をクリア（学習をリセット）
      gmm.Clear();
  }

  return 0;
}
