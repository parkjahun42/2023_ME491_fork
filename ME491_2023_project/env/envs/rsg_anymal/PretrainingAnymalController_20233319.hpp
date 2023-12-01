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
#include "math.h"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class PretrainingAnymalController_20233319 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    cage_ = reinterpret_cast<raisim::Visuals *>(world->getObject("cage"));
    box_ = reinterpret_cast<raisim::Box *>(world->getObject("opponent"));
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

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    opponent_gc_.setZero(gcDim_);
    opponent_gv_.setZero(gvDim_);
    opponent_gc_init_.setZero(gcDim_);

    cage2base_pos_xy_.setZero(2);
    opponent_cage2base_pos_xy_.setZero(2);

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 73;
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
      gc_init_.head(3) << cage_radius_ / 2 * std::cos(theta), cage_radius_ / 2 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }

    if(israndomizeBoxPosition) randomizeBoxPosition(world, theta);
    else{
      box_->setPosition(0,0,0.5);
      //    box_->setPosition(1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5);
      box_->setOrientation(cos(theta / 2), 0, 0, sin(theta / 2));
      opponent_gc_init_.head(3) << 0,0,0;//1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
    }
//    if(israndomizeBoxVelocity) randomizeBoxVelocity(world);

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
    cage2base_pos_ = gc_.head(3);
    cage2base_pos_xy_ = cage2base_pos_.head(2);

    //update opponent robot's data
    raisim::Mat<3, 3> opponent_rot;
    raisim::Vec<3> opponent_pos, opponent_linearVel, opponent_angularVel;
    Eigen::Vector3d opponentLinearVel2Body_, opponentAngularVel2Body_;
    box_->getPosition(opponent_pos);
    box_->getLinearVelocity(opponent_linearVel);
    box_->getAngularVelocity(opponent_angularVel);

    opponent_gc_.head(3) = opponent_pos.e();
//    opponent_gc_.head(3) = box_->getPosition();

    opponent_rot = box_->getOrientation();
    opponentLinearVel2Body_ =  rot.e().transpose() * opponent_linearVel.e();
    opponentAngularVel2Body_ = rot.e().transpose() * opponent_angularVel.e();

//    anymal_->getContacts()


//    raisim::Vec<4> opponent_quat;

//    opponent_quat[0] = opponent_gc_[3];
//    opponent_quat[1] = opponent_gc_[4];
//    opponent_quat[2] = opponent_gc_[5];
//    opponent_quat[3] = opponent_gc_[6];

    //update opponent cage's data
    opponent_cage2base_pos_xy_ = opponent_gc_.head(2);

    obDouble_ << bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6.
                 gc_[2], /// body pose 1
                 rot.e().row(2).transpose(), /// body orientation 3
                 gc_.tail(12), /// joint angles 12
                 gv_.tail(12), /// joint velocity 12
                 previousAction_, prepreviousAction_, /// previous action 24
                 cage2base_pos_xy_.norm(), /// cage2base xy position 1

                  //opponent related data
                  opponent_gc_.head(3) - gc_.head(3), /// Relative opponent player xyz position 3
                  opponentLinearVel2Body_, /// opponent player linear velocity 3
                  opponentAngularVel2Body_, /// opponent player angular velocity 3
                  opponent_rot.e().row(2).transpose(), /// opponent player orientation 3
                  opponent_cage2base_pos_xy_.norm(),
                  box_->getMass(); /// opponent cage2base xy position 1

  }

  inline void recordReward(Reward *rewards) {
    double poseError = (gc_.head(2) - opponent_gc_.head(2)).squaredNorm();

    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d targetVector = rot.e().transpose() * (opponent_gc_.head(3) - gc_.head(3)) / (opponent_gc_.head(3) - gc_.head(3)).norm();

    rewMove2Opponent_ = exp(-poseError / 0.25);
    rewForwardVel_ = exp(-(bodyLinearVel_.head(2) - targetVector.head(2)).squaredNorm()*0.5 / 0.25); // std::min(0.5, (gv_.head(2) - targetVector.head(2)*0.5).norm());
    rewTorque_ = anymal_->getGeneralizedForce().squaredNorm();
    rewTakeGoodPose = std::max((cage2base_pos_xy_.norm() - opponent_cage2base_pos_xy_.norm()), 0.0);
    rewOpponent2CageDist_ = ((opponent_cage2base_pos_xy_).norm()-opponent_gc_init_.head(2).norm());
    rewPushOpponentOff_ = (opponent_gc_.head(2).norm() > cage_radius_) ? 1.0 : 0.0;
    rewBaseMotion_ = (0.8 * bodyLinearVel_[2] * bodyLinearVel_[2] + 0.2 * fabs(bodyAngularVel_[0]) + 0.2 * fabs(bodyAngularVel_[1]));
    rewJointPosition = (gc_.tail(nJoints_) - gc_init_.tail(nJoints_)).norm();
    //opponent_gc_init_.head(2) << opponent_cage2base_pos_xy_;

    rewards->record("forwardVel", rewForwardVel_);
    rewards->record("move2Opponent",rewMove2Opponent_);
    rewards->record("torque", rewTorque_);
    rewards->record("takeGoodPose", rewTakeGoodPose);
    rewards->record("opponent2CageDist", rewOpponent2CageDist_);
    rewards->record("pushOpponentOff", rewPushOpponentOff_);
    rewards->record("baseMotion", rewBaseMotion_);
    rewards->record("jointPosition", rewJointPosition);

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

  void setBox(raisim::Box *box) {
    box_ = box;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  void setCageRadius(const double &cage_radius) {
    cage_radius_ = cage_radius;
  }

  inline int isTerminalState(raisim::World *world) {
    for (auto &contact: anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() && contact.getPairObjectIndex() == world->getObject("ground")->getIndexInWorld()) {
        return 1;
      }
      if(contact.getPairObjectIndex() == world->getObject("ground")->getIndexInWorld() &&
          contact.getlocalBodyIndex() == anymal_->getBodyIdx("base")) {
        return 1;
      }
    }
    if (gc_.head(2).norm() > cage_radius_) {
      return 2;
    }
    if(opponent_gc_.head(2).norm() > cage_radius_) {
      return 3;
    }

    return 0;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }

  void randomizeBoxPosition(raisim::World *world,  double theta){
    auto oppositeAngle = (uniDist_(gen_) + 0.5) * M_PI;
    double radius = 1.5;
//    box_->setPosition(0,0,0.5);
    if(isBoxPosCurriculum){
      radius = 0.0 + cage_radius_ / 2 * iter_ / 1000;
      box_->setMass(1.0 + 35.0 * iter_ / 10000);
    }
    else radius = 1.5;
    box_->setPosition(radius * std::cos(theta + oppositeAngle), radius * std::sin(theta + oppositeAngle), 0.5);
    box_->setOrientation(cos(theta / 2), 0, 0, sin(theta / 2));
    opponent_gc_init_.head(3) << radius * std::cos(theta + oppositeAngle), radius * std::sin(theta + oppositeAngle), 0.0;//1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
  }

  void curriculumUpdate(int iter){
    iter_ = iter;
  }

    //RandomizeRelated
  bool israndomizeBoxPosition = true;
  bool israndomizeBoxVelocity = true;
  bool isBoxPosCurriculum = true;
  bool isCageRadiusCurriculum = true;

 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_;
  raisim::Box *box_;
  raisim::Visuals *cage_;
  double cage_radius_ = 3.0;
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
  Eigen::VectorXd opponent_gc_, opponent_gv_, opponent_gc_init_;
  Eigen::Vector3d opponent_bodyLinearVel_, opponent_bodyAngularVel_;
  Eigen::VectorXd opponent_cage2base_pos_xy_;

  //Train Related
  int iter_ = 0;

  //RewardRelated
  double rewForwardVel_ = 0., rewMove2Opponent_ = 0., rewTorque_ = 0., rewTakeGoodPose = 0., rewOpponent2CageDist_ = 0., rewPushOpponentOff_ = 0., rewBaseMotion_=0.,
          rewJointPosition = 0.;



  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::PretrainingAnymalController_20233319::gen_;
thread_local std::uniform_real_distribution<double> raisim::PretrainingAnymalController_20233319::uniDist_(0., 1.);
}