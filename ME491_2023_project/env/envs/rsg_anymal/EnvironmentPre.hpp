// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

#include "PretrainingAnymalController_20233319.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    auto* robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    auto* box = world_.addBox(0.7, 0.7, 0.7, 0.38);
    box->setName("opponent");
    robot->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
//    controller_.setBox(box);
    controller_.setCageRadius(cage_radius_);
//    controller_.setCfg(cfg);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      server_->focusOn(robot);
      cage_ = server_->addVisualCylinder("cage", cage_radius_, 0.05);
      cage_->setPosition(0,0,0);
      cage_->setCylinderSize(cage_radius_, 0.05);
    }
  }

  void init() {}

  void reset() {
    if(controller_.isCageRadiusCurriculum) {
      cage_radius_ = 2.0 + 1.0 * std::min(1.0, (double)(iter_ / controller_.cageRadiusCurriculumIter));
      controller_.setCageRadius(cage_radius_);
    }
    else cage_radius_ = 3.0;

    auto theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    controller_.advance(&world_, action);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservation(&world_);
    controller_.recordReward(&rewards_);
    return rewards_.sum();
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
  }

  bool isTerminalState(float &terminalReward) {
    int terminalState = controller_.isTerminalState(&world_);

    if(terminalState == 1 || terminalState == 2) {
      terminalReward = terminalRewardCoeff_;
      return true;
    }
    else if(terminalState == 3) {
      terminalReward = 0.f;
      return true;
    }

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate(int iter) {
    iter_ = iter;
    controller_.curriculumUpdate(iter_);
  };

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return &world_; }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;
  PretrainingAnymalController_20233319 controller_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;

  raisim::Visuals *cage_;

  double cage_radius_ = 3.0;
  int iter_ = 0;

};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}

