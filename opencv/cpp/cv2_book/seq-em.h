/*! \file    seq-em.h
    \brief   ���祬����ʬ�ۤ��Ф���EM���르�ꥺ��(�إå�) */
//------------------------------------------------------------------------------
#ifndef seq_em_h
#define seq_em_h
//------------------------------------------------------------------------------
#include <opencv/ml.h>
#include <opencv2/legacy/legacy.hpp>
// NOTE: necessary for CvEM as it is legacy (there will be a new implementation)
//------------------------------------------------------------------------------
// TGMM2DParams::OMReuseFlag �ϰʲ����Ȥ߹�碌
static const char OM_REUSE_MEANS    (0x01);
static const char OM_REUSE_COVS     (0x02);
static const char OM_REUSE_WEIGHTS  (0x04);
//------------------------------------------------------------------------------

// TGMM2D �Υѥ�᥿
struct TGMM2DParams
{
  int NModels;
  int NSamples;

  int CovMatType;
  int StartStep;
  CvTermCriteria  TermCriteria;

  bool OnlineMode;  // ����饤��⡼��(EM ������Υѥ�᥿���Ѥ���)
  int  OMReuseFlag;  // ����饤��⡼�ɤǺ����Ѥ���ѥ�᥿(������������������)
  int  OMMaxIter;  // ����饤��⡼�ɤǤ� EM �κ���ȿ���� (max_iter)

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

// CvEM �Υ�åѡ����饹
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

  // ��ħ��(s1,s2)�Υ��饹����ꤷ����ǥå������֤�
  // probs: NULL �Ǥʤ����, ��ħ�̤��ƥ��饹��°�����Ψ���Ǽ����
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
