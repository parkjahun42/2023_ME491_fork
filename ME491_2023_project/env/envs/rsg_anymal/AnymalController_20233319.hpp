// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_20233319 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    opponent_anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));

    cage_ = reinterpret_cast<raisim::Visuals *>(world->getObject("cage"));
    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);
    previousAction_.setZero(nJoints_);
    prepreviousAction_.setZero(nJoints_);

    // initialize opponent robot's data
    opponent_gc_.setZero(gcDim_);
    opponent_gv_.setZero(gvDim_);

    // initialize opponent robot's data
    cage2base_pos_xy_.setZero(2);


    

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 58;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling

    //Stack Action History
    prepreviousAction_ = previousAction_;
    previousAction_ = action.cast<double>();

    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  inline bool reset(raisim::World *world, double theta) {
    if (playerNum_ == 0) {
      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

  inline void updateObservation(raisim::World *world) {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    //update cage's data
    Eigen::Vector3d cage2base_pos_;
    cage2base_pos_ = gc_.head(3) - cage_->getPosition();
    cage2base_pos_xy_ = cage2base_pos_.head(2);

    //update opponent robot's data
    opponent_anymal_->getState(opponent_gc_, opponent_gv_);
    raisim::Vec<4> opponent_quat;
    raisim::Mat<3, 3> opponent_rot;
    opponent_quat[0] = opponent_gc_[3];
    opponent_quat[1] = opponent_gc_[4];
    opponent_quat[2] = opponent_gc_[5];
    opponent_quat[3] = opponent_gc_[6];
    raisim::quatToRotMat(opponent_quat, opponent_rot);
    opponent_bodyLinearVel_ = opponent_rot.e().transpose() * opponent_gv_.segment(0, 3);
    opponent_bodyAngularVel_ = opponent_rot.e().transpose() * opponent_gv_.segment(3, 3);

    //update opponent cage's data
    opponent_cage2base_pos_xy_ = opponent_gc_.head(2) - cage_->getPosition().head(2);

    obDouble_ << bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6.
                 gc_[2], /// body pose 1
                 rot.e().row(2).transpose(), /// body orientation 3
                 gc_.tail(12), /// joint angles 12
                 gv_.tail(12), /// joint velocity 12
                 previousAction_, prepreviousAction_; /// previous action 24
//                 cage2base_pos_xy_; /// cage2base xy position 2
  }

  inline void recordReward(Reward *rewards) {
    rewards->record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards->record("forwardVel", std::min(4.0, bodyLinearVel_[0]));
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    for (auto &contact: anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_, *opponent_anymal_;
  raisim::Visuals *cage_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd cage2base_pos_xy_;
  Eigen::VectorXd previousAction_, prepreviousAction_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;

  //opponent robot's data
  Eigen::VectorXd opponent_gc_, opponent_gv_;
  Eigen::Vector3d opponent_bodyLinearVel_, opponent_bodyAngularVel_;
  Eigen::VectorXd opponent_cage2base_pos_xy_;
};

class OpponentController_20233319 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    opponent_anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(opponentName_));

    cage_ = reinterpret_cast<raisim::Visuals *>(world->getObject("cage"));
    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);
    previousAction_.setZero(nJoints_);
    prepreviousAction_.setZero(nJoints_);

    // initialize opponent robot's data
    opponent_gc_.setZero(gcDim_);
    opponent_gv_.setZero(gvDim_);

    // initialize opponent robot's data
    cage2base_pos_xy_.setZero(2);




    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 58;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling

    //Stack Action History
    prepreviousAction_ = previousAction_;
    previousAction_ = action.cast<double>();

    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
//    pTarget_.tail(nJoints_) = pTarget12_;
    pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  inline bool reset(raisim::World *world, double theta) {
    if (playerNum_ == 0) {
      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

  inline void updateObservation(raisim::World *world) {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    //update cage's data
//    cage2base_pos_xy_ = gc_.head(2) - cage_->getPosition().head(2);

    //update opponent robot's data
    opponent_anymal_->getState(opponent_gc_, opponent_gv_);
    raisim::Vec<4> opponent_quat;
    raisim::Mat<3, 3> opponent_rot;
    opponent_quat[0] = opponent_gc_[3];
    opponent_quat[1] = opponent_gc_[4];
    opponent_quat[2] = opponent_gc_[5];
    opponent_quat[3] = opponent_gc_[6];
    raisim::quatToRotMat(opponent_quat, opponent_rot);
    opponent_bodyLinearVel_ = opponent_rot.e().transpose() * opponent_gv_.segment(0, 3);
    opponent_bodyAngularVel_ = opponent_rot.e().transpose() * opponent_gv_.segment(3, 3);

    //update opponent cage's data
//    opponent_cage2base_pos_xy_ = opponent_gc_.head(2) - cage_->getPosition().head(2);

    obDouble_ << bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6.
                 gc_[2], /// body pose 1
                 rot.e().row(2).transpose(), /// body orientation 3
                 gc_.tail(12), /// joint angles 12
                 gv_.tail(12), /// joint velocity 12
                 previousAction_, prepreviousAction_; /// previous action 24
//                 cage2base_pos_xy_; /// cage2base xy position 2
  }

  inline void recordReward(Reward *rewards) {
    rewards->record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards->record("forwardVel", std::min(4.0, bodyLinearVel_[0]));
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    for (auto &contact: anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }
    }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_, *opponent_anymal_;
  raisim::Visuals *cage_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd cage2base_pos_xy_;
  Eigen::VectorXd previousAction_, prepreviousAction_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;

  //opponent robot's data
  Eigen::VectorXd opponent_gc_, opponent_gv_;
  Eigen::Vector3d opponent_bodyLinearVel_, opponent_bodyAngularVel_;
  Eigen::VectorXd opponent_cage2base_pos_xy_;
};

}


