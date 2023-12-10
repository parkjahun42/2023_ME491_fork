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

#include PLAYER1_HEADER_FILE_TO_INCLUDE
#include PLAYER2_HEADER_FILE_TO_INCLUDE

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int mode, int init_mode) :
      visualizable_(visualizable) {
    /// add objects
    auto* robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robot->setName(PLAYER1_NAME);
    controller_.setName(PLAYER1_NAME);
    controller_.setOpponentName(PLAYER2_NAME);
    controller_.setPlayerNum(0);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    raisim::ArticulatedSystem* opponent_robot;

    if(init_mode == 0){
      opponent_robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
    }
    else if(init_mode == -1){
      opponent_robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red_big.urdf");
    }

    opponent_robot->setName(PLAYER2_NAME);
    opponent_controller_.setName(PLAYER2_NAME);
    opponent_controller_.setOpponentName(PLAYER1_NAME);
    opponent_controller_.setPlayerNum(1);
    opponent_controller_.setOpponentMode(mode);
    opponent_controller_.setOpponentInitMode(init_mode);
    opponent_robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);


    controller_.setCageRadius(cage_radius_);
    opponent_controller_.setCageRadius(cage_radius_);

    auto* ground = world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    opponent_controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])
    READ_YAML(double, episode_time_, cfg["episode_time"])

    controller_.setEpisodeTime(episode_time_);
    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);



    /// visualize if it is the first environment
    if (visualizable_) {

      controller_.visualizable_ = visualizable_;
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
    if(opponent_controller_.isCageRadiusCurriculum) {
      cage_radius_ = 2.0 + 1.0 * std::min(1.0, ((double)iter_ / (double)opponent_controller_.cageRadiusCurriculumIter));
      controller_.setCageRadius(cage_radius_);
      opponent_controller_.setCageRadius(cage_radius_);
    }
    else cage_radius_ = 3.0;

    auto theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);
    opponent_controller_.reset(&world_, theta);
    controller_.curriculumLevel = opponent_controller_.curriculumLevel;
    controller_.setOpponentGcInit(opponent_controller_.getGcInit());
    opponent_controller_.setOpponentGcInit(controller_.getGcInit());
    timer_ = 0;
  }

  float step(const Eigen::Ref<EigenVec> &action, const Eigen::Ref<EigenVec> &opponent_action) {
    timer_ += 1;
    controller_.advance(&world_, action);
    opponent_controller_.advance(&world_, opponent_action);

    if(visualizable_){
      cage_->setCylinderSize(cage_radius_, 0.05);
      commandSphere_->setPosition(opponent_controller_.globalCommandPoint);
    }

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {

      if(opponent_controller_.israndomizeOpponentExtForce) {
        Eigen::Vector3d opponentPos = opponent_controller_.getGc().head(3);
        opponent_controller_.anymal_->setExternalForce(0, {opponentPos(0), opponentPos(1), 0.}, opponent_controller_.opponentExternalForce_);
      }

      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservation(&world_);
    controller_.updateCurrentTime(control_dt_);
    controller_.recordReward(&rewards_);
    opponent_controller_.updateObservation(&world_);
    opponent_controller_.updateCurrentTime(control_dt_);
     return rewards_.sum();
  }

  //TODO: modify observe and isTerminalState function
  void observe(Eigen::Ref<EigenVec> ob, Eigen::Ref<EigenVec> opponent_ob) {
    controller_.updateObservation(&world_);
    opponent_controller_.updateObservation(&world_);
    ob = controller_.getObservation().cast<float>();
    opponent_ob = opponent_controller_.getObservation().cast<float>();

  }

  bool player1_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER1_NAME));
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

  bool player2_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER2_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
      if(opponent_mode_ != 4) {
        if (contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
            contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
          if(visualizable_) std::cout << "base contact!" << std::endl;
          return true;
        }
      }
    }
    /// get out of the cage
    int gcDim = anymal->getGeneralizedCoordinateDim();
    Eigen::VectorXd gc;
    gc.setZero(gcDim);
    gc = anymal->getGeneralizedCoordinate().e();
    if (gc.head(2).norm() > cage_radius_) {
      if(visualizable_) std::cout << "out!" << std::endl;
      return true;
    }
    return false;
  }

  bool isTerminalState(float &terminalReward) {
//    if(controller_.isTerminalState(&world_)) {
//      terminalReward = terminalRewardCoeff_;
//      return true;
//    }
//    terminalReward = 0.f;
//    return false;
    if (player1_die() && player2_die()) {
      terminalReward = 0.f;
      return true;
    }

    if (timer_ > 10 * 100) {
      if(visualizable_) std::cout << "timeout" << std::endl;
      terminalReward = -4.f;
      return true;
    }


    if (!player1_die() && player2_die()) {
      if(visualizable_) std::cout << "player2 die" << std::endl;
      terminalReward = -terminalRewardCoeff_;
      opponent_controller_.curriculumLevel += 1;
      opponent_controller_.curriculumLevelUpdate();
      return true;
    }

    if (player1_die() && !player2_die()) {
        if(visualizable_) std::cout << "player1 die" << std::endl;
      terminalReward = terminalRewardCoeff_;
      if(timer_ < 3 * 100) {
        opponent_controller_.curriculumLevel -= 1;
        opponent_controller_.curriculumLevelUpdate();
      }
      return true;
    }
    return false;
  }

  void curriculumUpdate(int iter) {
    iter_ = iter;
    controller_.curriculumUpdate(iter_);
    opponent_controller_.curriculumUpdate(iter_);
  };

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  void setOpponentMode(int mode) {
    opponent_mode_ = mode;
    opponent_controller_.setOpponentMode(mode);
  }

  int getCurriculumLevel(int level) {
    return opponent_controller_.curriculumLevel;
  }

  int getObDim() { return controller_.getObDim(); }

  int getOpponentObDim() { return opponent_controller_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  int getCurriculumLevel() { return opponent_controller_.curriculumLevel; }

  void setCurriculumLevelZero() { opponent_controller_.curriculumLevel = 0; }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return &world_; }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  int timer_ = 0;
  int opponent_mode_ = 0;
  bool visualizable_ = false;
  double terminalRewardCoeff_ = -10.;

  int gameCount_ = 0;

  PLAYER1_CONTROLLER controller_;
  PLAYER2_CONTROLLER opponent_controller_;
  raisim::World world_;
  raisim::Reward rewards_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  double episode_time_ = 10.0 ;
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

