/*! \file    seq-em.h
    \brief   混合ガウス分布に対するEMアルゴリズム(ヘッダ) */
//------------------------------------------------------------------------------
#ifndef seq_em_h
#define seq_em_h
//------------------------------------------------------------------------------
#include <opencv/ml.h>
#include <opencv2/legacy/legacy.hpp>
// NOTE: necessary for CvEM as it is legacy (there will be a new implementation)
//------------------------------------------------------------------------------
// TGMM2DParams::OMReuseFlag は以下の組み合わせ
static const char OM_REUSE_MEANS    (0x01);
static const char OM_REUSE_COVS     (0x02);
static const char OM_REUSE_WEIGHTS  (0x04);
//------------------------------------------------------------------------------

// TGMM2D のパラメタ
struct TGMM2DParams
{
  int NModels;
  int NSamples;

  int CovMatType;
  int StartStep;
  CvTermCriteria  TermCriteria;

  bool OnlineMode;  // オンラインモード(EM で前回のパラメタを用いる)
  int  OMReuseFlag;  // オンラインモードで再利用するパラメタ(上で定義した定数を使用)
  int  OMMaxIter;  // オンラインモードでの EM の最大反復数 (max_iter)

  TGMM2DParams()
    :
      NModels       (2),
      NSamples      (5000),
      CovMatType    (CvEM::COV_MAT_GENERIC),
                    // or COV_MAT_DIAGONAL or COV_MAT_SPHERICAL
      StartStep     (CvEM::START_AUTO_STEP),
      OnlineMode    (true),
      OMReuseFlag   (OM_REUSE_MEANS | OM_REUSE_WEIGHTS),
      OMMaxIter     (2)
    {
      TermCriteria.max_iter= 10;
      TermCriteria.epsilon= 0.1;
      TermCriteria.type= CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    }
};

// CvEM のラッパークラス
class TGMM2D
{
public:

  TGMM2D();
  ~TGMM2D()  {}

  void Clear();

  void AddToData(const float &s1, const float &s2);

  bool EMExecutable() const
    {return !params_.OnlineMode || ncurrsmpl_>=params_.NSamples;}
  void TrainByEM();

  // 特徴量(s1,s2)のクラスを推定しインデックスを返す
  // probs: NULL でなければ, 特徴量が各クラスに属する確率を格納する
  int  Predict(const float &s1, const float &s2, cv::Mat *probs=NULL);

  const cv::Mat& GetSamples() const {return samples_;}
  int  NumOfSamples() const {return ncurrsmpl_;}
  const CvEM& GetEMModel() const {return *curr_em_model_;}

  const TGMM2DParams& Params() const {return params_;}
  TGMM2DParams& SetParams()  {return params_;}

private:

  TGMM2D(const TGMM2D&);
  const TGMM2D& operator=(const TGMM2D&);

  TGMM2DParams  params_;

  bool first_em_;
  int ncurrsmpl_, sample_idx_;

  cv::Mat samples_;
  CvEM  em_model1_, em_model2_;
  CvEM  *curr_em_model_, *prev_em_model_;

  float tmp_sample_entity_[2];
  cv::Mat tmp_sample_;
  cv::Mat tmp_probs_;
};

#endif // seq_em_h
