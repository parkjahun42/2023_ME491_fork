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

#include TRAINING_HEADER_FILE_TO_INCLUDE

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add objects
    auto* robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robot->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    controller_.setOpponentName("opponent");
    controller_.setPlayerNum(0);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    auto* opponent_robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    opponent_robot->setName("opponent");
    opponent_controller_.setName("opponent");
    opponent_controller_.setOpponentName(PLAYER_NAME);
    opponent_controller_.setPlayerNum(1);
    opponent_robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    world_.addGround();

    controller_.create(&world_);
    opponent_controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      server_->focusOn(robot);
      auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
      cage->setPosition(0,0,0);
    }
  }

  void init() {}

  void reset() {
    auto theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);
    opponent_controller_.reset(&world_, theta);
  }

  float step(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &opponent_action) {
    controller_.advance(&world_, action);
    opponent_controller_.advance(&world_, opponent_action);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservation(&world_);
    controller_.recordReward(&rewards_);
    opponent_controller_.updateObservation(&world_);
     return rewards_.sum();
  }

  //TODO: modify observe and isTerminalState function
  void observe(Eigen::Ref<EigenVec> ob, Eigen::Ref<EigenVec> opponent_ob) {
    controller_.updateObservation(&world_);
    opponent_controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
    opponent_ob = opponent_controller_.getObservation().cast<float>();
  }

  bool isTerminalState(float &terminalReward) {
    if(controller_.isTerminalState(&world_)) {
      terminalReward = terminalRewardCoeff_;
      return true;
    }
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {};

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
  TRAINING_CONTROLLER controller_;
  OpponentController_20233319 opponent_controller_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}

