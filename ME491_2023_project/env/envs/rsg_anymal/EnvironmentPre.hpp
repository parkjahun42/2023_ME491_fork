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


    robot->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);

    controller_.setCageRadius(cage_radius_);

//    controller_.setCfg(cfg);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, episode_time_, cfg["episode_time"])
    controller_.setEpisodeTime(episode_time_);
    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

      /// indices of links that should not make contact with ground
    footIndices_.insert(robot->getBodyIdx("LF_SHANK"));
    footIndices_.insert(robot->getBodyIdx("RF_SHANK"));
    footIndices_.insert(robot->getBodyIdx("LH_SHANK"));
    footIndices_.insert(robot->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      server_->focusOn(robot);
      cage_ = server_->addVisualCylinder("cage", cage_radius_, 0.05);
      commandSphere_ = server_->addVisualSphere("commandSphere", 0.2, 0.0, 0.0, 1.0, 0.5);
      cage_->setPosition(0,0,0);
      cage_->setCylinderSize(cage_radius_, 0.05);
    }
  }

  void init() {}

  void reset() {
    if(controller_.isCageRadiusCurriculum) {
      cage_radius_ = 2.0 + 1.0 * std::min(1.0, ((double)iter_ / (double)controller_.cageRadiusCurriculumIter));
      controller_.setCageRadius(cage_radius_);
    }
    else{
      cage_radius_ = default_cage_radius_;
      controller_.setCageRadius(cage_radius_);
    }

    auto theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);
    timer_ = 0;
//    commandSpheres_[0]->setPosition(0,0,2.5);
    if(visualizable_){
      cage_->setPosition(0,0,0);
      cage_->setCylinderSize(cage_radius_, 0.05);
      commandSphere_->setPosition(controller_.globalCommandPoint);
    }
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    timer_ += 1;
    controller_.advance(&world_, action);
    if(visualizable_){
      commandSphere_->setPosition(controller_.globalCommandPoint);
      if(controller_.continuousGoalCount > 200) commandSphere_->setColor(0.0, 1.0, 0.0, 0.5);
      else commandSphere_->setColor(0.0, 0.0, 1.0, 0.5);
    }
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      raisim::Vec<3> opponentPos;
      controller_.anymal_->getPosition(0, opponentPos);
      controller_.anymal_->setExternalForce(0, opponentPos, controller_.boxExternalForce_);

      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservation(&world_);
    controller_.updateCurrentTime(control_dt_);
    controller_.recordReward(&rewards_);
    return rewards_.sum();
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
  }

    bool player1_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
      if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
          contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
        return true;
      }
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld()) {
        return true;
      }
    }
    /// get out of the cage
    int gcDim = anymal->getGeneralizedCoordinateDim();
    Eigen::VectorXd gc;
    gc.setZero(gcDim);
    gc = anymal->getGeneralizedCoordinate().e();
    if (gc.head(2).norm() > cage_radius_) {
      return true;
    }
    return false;
  }

  bool player2_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject("opponent"));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
      if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
          contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
        return true;
      }
    }
    /// get out of the cage
    int gcDim = anymal->getGeneralizedCoordinateDim();
    Eigen::VectorXd gc;
    gc.setZero(gcDim);
    gc = anymal->getGeneralizedCoordinate().e();
    if (gc.head(2).norm() > cage_radius_) {
      return true;
    }
    return false;
  }

  bool isTerminalState(float &terminalReward) {

//    if (player1_die() && player2_die()) {
//      terminalReward = 0.f;
//      return true;
//    }

    if (timer_ > 10. * 1 / control_dt_) {
      controller_.checkcommandPointSuccess(true);
      if(controller_.commandSuccessCount > 0){
        terminalReward = controller_.commandSuccessCount * 10.f;
      }
      else terminalReward = -4.f;
      return true;
    }


//    if (!player1_die() && player2_die()) {
//      terminalReward = 0.f;
//      return true;
//    }

    if (player1_die()) {
      terminalReward = terminalRewardCoeff_;
      return true;
    }
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
  std::set<size_t> footIndices_;
  int timer_ = 0;
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;
  PretrainingAnymalController_20233319 controller_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  double episode_time_ = 10.;
  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;

  raisim::Visuals *cage_, *commandSphere_;
  std::vector<raisim::Visuals*> commandSpheres_;

  double cage_radius_ = 3.0, default_cage_radius_ = 3.0;
  int iter_ = 0;

};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}

