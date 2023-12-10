//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {
    Yaml::Parse(cfg_, cfg);

    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();

    environments_.reserve(num_envs_);
    rewardInformation_.reserve(num_envs_);

    mode_.setZero(5); // PD, ME, Sphere, Mugisung, Box
    mode_ << 0.75, 0.0, 0.05, 0.0, 0.2;
    curriculumLevel_.setZero(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      if(i < mode_[0] * num_envs_) environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, 0, 0));
      else if(i < (mode_[0] + mode_[1]) * num_envs_) environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, 1, 0));
      else if(i < (mode_[0] + mode_[1] + mode_[2]) * num_envs_) environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, 2, 0));
      else if(i < (mode_[0] + mode_[1] + mode_[2] + mode_[3]) * num_envs_) environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, 3, 0));
      else if(i < (mode_[0] + mode_[1] + mode_[2] + mode_[3] + mode_[4]) * num_envs_) environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, 4, -1));

//      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
      rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
    }

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    opponentObDim_ = environments_[0]->getOpponentObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      opponentObMean_.setZero(opponentObDim_);
      opponentObVar_.setOnes(opponentObDim_);
      opponentOb2Mean_.setZero(opponentObDim_);
      opponentOb2Var_.setOnes(opponentObDim_);
      opponentOb3Mean_.setZero(opponentObDim_);
      opponentOb3Var_.setOnes(opponentObDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, Eigen::Ref<EigenRowMajorMat> &opponent_ob, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i), opponent_ob.row(i));


    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, opponent_ob, updateStatistics);
  }


  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenRowMajorMat> &opponent_action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, opponent_action, reward, done);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }
  void setOpponentObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    opponentObMean_ = mean; opponentObVar_ = var;}
  void setOpponentObStatistics2(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    opponentOb2Mean_ = mean; opponentOb2Var_ = var;}
  void setOpponentObStatistics3(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    opponentOb3Mean_ = mean; opponentOb3Var_ = var;}

  void setSeed(int seed) {
    int seed_inc = seed;

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      environments_[i]->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setMode() {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++) {
      if(i < mode_[0] * num_envs_) environments_[i]->setOpponentMode(0);
      else if(i < (mode_[0] + mode_[1]) * num_envs_) environments_[i]->setOpponentMode(1);
      else if(i < (mode_[0] + mode_[1] + mode_[2]) * num_envs_) environments_[i]->setOpponentMode(2);
      else if(i < (mode_[0] + mode_[1] + mode_[2] + mode_[3]) * num_envs_) environments_[i]->setOpponentMode(3);
      else if(i < (mode_[0] + mode_[1] + mode_[2] + mode_[3] + mode_[4]) * num_envs_) environments_[i]->setOpponentMode(4);
    }
  }



  void checkCurriculum() {
//    std::cout << curriculumLevel_.head((int)(mode_[0] * num_envs_)).mean() << std::endl;
    if(modeLevel == 0) {
      if (curriculumLevel_.head((int)(mode_[0] * num_envs_)).mean() > 100){
        if(mode_[0] < 0.21)
        {
          #pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++) {
            if(i < (int)(mode_[0] * num_envs_)) environments_[i]->setCurriculumLevelZero();
          }
          mode_ << 0.0, 0.0, 0.8, 0.0, 0.2;
          modeLevel++;
          setMode();
        }
        else {
          #pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++) {
            if (i < (int) (mode_[0] * num_envs_)) environments_[i]->setCurriculumLevelZero();
          }
          mode_[0] -= 0.1;
          mode_[2] += 0.1;
//        environments_[0]-> visualizable_ = false;
//        environments_[(int)((mode_[0] + mode_[1])*num_envs_ +1)]-> visualizable_ = true;


          setMode();
        }
      }

//      if (curriculumLevel_.head((int)(mode_[0] * num_envs_)).mean() > 150){
//        mode_ << 0.0, 0.0, 0.8, 0.0, 0.2;
////        environments_[0]-> visualizable_ = false;
////        environments_[(int)((mode_[0] + mode_[1])*num_envs_ +1)]-> visualizable_ = true;
//        modeLevel++;
//        setMode();
//      }
    }
    else if(modeLevel == 1)
    {
//      std::cout << curriculumLevel_.segment((int)((mode_[0] + mode_[1]) * num_envs_), (int)((mode_[2]) * num_envs_)).mean() << std::endl;
      if (curriculumLevel_.segment((int)((mode_[0] + mode_[1]) * num_envs_), (int)((mode_[2]) * num_envs_)).mean() > 100){ //&& curriculumLevel_.segment((int)((mode_[0] + mode_[1] + mode_[2]) * num_envs_), (int)((mode_[3]) * num_envs_)).mean() > 150){
        #pragma omp parallel for schedule(auto)
        for (int i = 0; i < num_envs_; i++) {
          if(i < (int)((mode_[0]+mode_[1]+mode_[2]) * num_envs_)) environments_[i]->setCurriculumLevelZero();
        }
        mode_ << 0.0, 0.1, 0.6, 0.1, 0.2;
        modeLevel++;
        setMode();
      }
    }
    else if(modeLevel == 2)
    {
      if (curriculumLevel_.segment((int)((mode_[0]) * num_envs_), (int)((mode_[1]) * num_envs_)).mean() > 100 && curriculumLevel_.segment((int)((mode_[0] + mode_[1]) * num_envs_), (int)((mode_[2]) * num_envs_)).mean() > 100 && curriculumLevel_.segment((int)((mode_[0] + mode_[1] + mode_[2]) * num_envs_), (int)((mode_[3]) * num_envs_)).mean() > 100){
        mode_ << 0.0, 0.5, 0.2, 0.1, 0.2;
//        environments_[0]-> visualizable_ = true;
//        environments_[(int)((0.4)*num_envs_ +1)]-> visualizable_ = false;
        modeLevel++;
        setMode();
      }
    }
    else if(modeLevel == 3)
    {
      if (curriculumLevel_.segment((int)((mode_[0]) * num_envs_), (int)((mode_[1]) * num_envs_)).mean() > 150){
        mode_ << 0.0, 0.6, 0.1, 0.1, 0.2;
        modeLevel++;
        setMode();
      }
    }

  }

  void getCurrLevel(Eigen::Ref<EigenVec> &currLevel) {
    currLevel = curriculumLevel_;
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getOpponentObDim() { return opponentObDim_; }
  int getActionDim() { return actionDim_; }
  int getModeNum() { return mode_.size(); }
  int getModeLevel() {return modeLevel; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate(int iter) {
    for (auto *env: environments_)
      env->curriculumUpdate(iter);
  }

  void modeUpdate(Eigen::Ref<EigenVec> &mode)
  {
    mode = mode_;
  }

  const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

 private:
  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, Eigen::Ref<EigenRowMajorMat> &opponent_ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for(int i=0; i<obDim_; i++)
        delta_[i] = delta_[i]*delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++) {
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>((obVar_ + epsilon).cwiseSqrt().transpose());
      if(i < (mode_[0] + mode_[1]) * num_envs_){
        opponent_ob.row(i) = (opponent_ob.row(i) - opponentObMean_.transpose()).template cwiseQuotient<>((opponentObVar_ + epsilon).cwiseSqrt().transpose());;
      }
      else if(i < (mode_[0] + mode_[1] + mode_[2]) * num_envs_) {
        opponent_ob.row(i) = (opponent_ob.row(i) - opponentOb2Mean_.transpose()).template cwiseQuotient<>((opponentOb2Var_ + epsilon).cwiseSqrt().transpose());
      }
      else if(i < (mode_[0] + mode_[1] + mode_[2] + mode_[3]) * num_envs_){
        opponent_ob.row(i) = (opponent_ob.row(i) - opponentOb3Mean_.transpose()).template cwiseQuotient<>((opponentOb3Var_ + epsilon).cwiseSqrt().transpose());
      }
    }
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                            Eigen::Ref<EigenRowMajorMat> &opponent_action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId), opponent_action.row(agentId));
    rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
    curriculumLevel_[agentId] = environments_[agentId]->getCurriculumLevel();
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::map<std::string, float>> rewardInformation_;

  int num_envs_ = 1;
  int obDim_ = 0, opponentObDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_, opponentObMean_, opponentOb2Mean_, opponentOb3Mean_;
  EigenVec obVar_, opponentObVar_, opponentOb2Var_, opponentOb3Var_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
  EigenVec mode_, curriculumLevel_;
  int modeLevel = 0;
};


class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
