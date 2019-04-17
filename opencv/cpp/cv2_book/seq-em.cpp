/*! \file    seq-em.cpp
    \brief   混合ガウス分布に対するEMアルゴリズム(実装) */
//------------------------------------------------------------------------------
#include "seq-em.h"
#include <iostream>
//------------------------------------------------------------------------------
using namespace std;

//==============================================================================
// class TGMM2D
//==============================================================================

TGMM2D::TGMM2D()
  :
    first_em_       (true),
    ncurrsmpl_      (0),
    sample_idx_     (0),
    curr_em_model_  (&em_model1_),
    prev_em_model_  (&em_model2_),
    tmp_sample_     (1, 2, CV_32FC1, tmp_sample_entity_)
{
}
//------------------------------------------------------------------------------

void TGMM2D::Clear()
{
  first_em_= true;
  ncurrsmpl_= 0;
  sample_idx_= 0;
}
//------------------------------------------------------------------------------

void TGMM2D::AddToData(const float &s1, const float &s2)
{
  samples_.create(params_.NSamples, 2, CV_32FC1);
  samples_.at<float>(sample_idx_,0)=s1;
  samples_.at<float>(sample_idx_,1)=s2;
  ++sample_idx_;
  ++ncurrsmpl_;
  if (sample_idx_>=params_.NSamples)  sample_idx_=0;
  if (ncurrsmpl_>params_.NSamples)  ncurrsmpl_=params_.NSamples;
}
//------------------------------------------------------------------------------

void TGMM2D::TrainByEM()
{
  if(!EMExecutable())  return;

  CvEMParams emparams;
  emparams.covs      = NULL;
  emparams.means     = NULL;
  emparams.weights   = NULL;
  emparams.probs     = NULL;
  emparams.nclusters = params_.NModels;
  emparams.cov_mat_type  = params_.CovMatType;
  emparams.start_step    = params_.StartStep;
  emparams.term_crit = params_.TermCriteria;

  swap(curr_em_model_,prev_em_model_);
  if (params_.OnlineMode && !first_em_)
  {
    emparams.term_crit.max_iter = params_.OMMaxIter;
    emparams.start_step = CvEM::START_E_STEP;
    if (params_.OMReuseFlag & OM_REUSE_MEANS)
      emparams.means   = prev_em_model_->get_means();
    if (params_.OMReuseFlag & OM_REUSE_COVS)
      emparams.covs    = prev_em_model_->get_covs();
    if (params_.OMReuseFlag & OM_REUSE_WEIGHTS)
      emparams.weights = prev_em_model_->get_weights();
  }

  // サンプルから学習
  CvMat samples2(samples_);
  curr_em_model_->train(&samples2, 0, emparams);
  first_em_= false;
}
//------------------------------------------------------------------------------

  // 特徴量(s1,s2)のクラスを推定しインデックスを返す
  // probs: NULL でなければ, 特徴量が各クラスに属する確率を格納する
int  TGMM2D::Predict(const float &s1, const float &s2, cv::Mat *probs)
{
  tmp_probs_.create(1, params_.NModels, CV_32FC1);

  tmp_sample_.at<float>(0,0)= s1;
  tmp_sample_.at<float>(0,1)= s2;
  CvMat tmp_sample2(tmp_sample_), tmp_probs2(tmp_probs_);
  int response= cvRound(curr_em_model_->predict(&tmp_sample2, &tmp_probs2));
  if (isnan(tmp_probs_.at<float>(0,response)))
  {
    cerr<<"FAILED TO PREDICT! PROBABILITY IS SET TO ZERO."<<endl;
    tmp_probs_= cv::Scalar(0.0f);
  }

  if (probs)  *probs= tmp_probs_;
  return response;
}
