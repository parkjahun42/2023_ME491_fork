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
class PretrainingAnymalController_20233319 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));

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
    opponent_gc_init_.setZero(gcDim_);
    opponent_gv_init_.setZero(gvDim_);
    opponent_pTarget_.setZero(gcDim_);

    opponent_base_collision_size_(0) = 0.537;
    opponent_base_collision_size_(1) = 0.27;
    opponent_base_collision_size_(2) = 0.24;

    // initialize opponent robot's data
    cage2base_pos_xy_.setZero(2);
    cage2base_pos_body_.setZero(3);
    opponent_cage2base_pos_xy_.setZero(2);
    globalCommandPoint.setZero();
    bodyCommandPoint.setZero();

    

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    opponent_gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 120;
    realObDim_ = 65;
    obDimTeacher_ = 74;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    obDoubleTeacher_.setZero(obDimTeacher_);

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

    if(israndomizeOpponentVelocity) randomizeBoxVelocity(world);

    commandPointUpdate();


    return true;
  }

  inline bool reset(raisim::World *world, double theta) {

    curriculumLevelUpdate();

    currentTime_ = 0.;
    boxExternalForce_.setZero();
    changeGoal = true;
    commandSuccessCount = 0;
    maxContinuousGoalCount = 0;
    commandPointCount = 0;

    auto oppositeAngle = uniDistCage_(gen_) * M_PI;
    auto randomYaw = uniDistCage_(gen_) * M_PI;
    double radius = 1.5;
//    opponent_->setPosition(0,0,0.5);`
    if(isOpponentPosCurriculum){
      double prob = uniDist_(gen_);

      if(prob > 0.5) radius = 0.0 + std::min(1.8, (cage_radius_ / 2) * std::min(1.0, ((double)iter_ / cageRadiusCurriculumIter)));
      else radius = 0.0 + uniDist_(gen_) * std::min(1.8, (cage_radius_ / 2) * std::min(1.0, ((double)iter_ / cageRadiusCurriculumIter)));
    }
    else radius = 1.5;

    gc_init_.head(3) << radius * std::cos(theta), radius * std::sin(theta), 0.5;
    double prob2 = uniDist_(gen_);

    if(prob2 > 0.7) gc_init_.segment(3, 4) << cos(theta / 2 + randomYaw), 0, 0, sin(theta / 2 + randomYaw);
    else gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);

    if(isOpponentMassCurriculum) opponentMassCurriculum();

    if(isOpponentBaseCollisionCurriculum) opponentBaseCollisionCurriculum();

    if(israndomizeOpponentVelocity) randomizeBoxVelocity(world);

    anymal_->setState(gc_init_, gv_init_);

    commandPointUpdate();
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
    cage2base_pos_body_ = rot.e().transpose() * (gc_.head(3));

    bodyCommandPoint = rot.e().transpose() * (globalCommandPoint - gc_.head(3));

    obDouble_ << bodyCommandPoint,
                 bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6.
                 gc_[2], /// body pose 1
                 rot.e().row(2).transpose(), /// body orientation 3
                 gc_.tail(12), /// joint angles 12
                 gv_.tail(12), /// joint velocity 12
                 previousAction_, prepreviousAction_, /// previous action 24
                 cage2base_pos_body_.head(2), /// cage2base xy position in body frame 2
                 cage2base_pos_xy_.norm(), /// cage2base xy position 1
                 cage_radius_,
                 Eigen::VectorXd::Ones(obDim_-realObDim_)*0.0;


//     obDoubleTeacher_ << bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6.
//                 gc_[2], /// body pose 1
//                 rot.e().row(2).transpose(), /// body orientation 3
//                 gc_.tail(12), /// joint angles 12
//                 gv_.tail(12), /// joint velocity 12
//                 previousAction_, prepreviousAction_, /// previous action 24
//                 cage2base_pos_xy_.norm(), /// cage2base xy position 1
//
//                //opponent related data
//                rot.e().transpose() * (opponent_gc_.head(3) - gc_.head(3)), /// Relative opponent player xyz position 3
//                opponent_bodyLinearVel_, /// opponent player linear velocity 3
//                opponent_bodyAngularVel_, /// opponent player angular velocity 3
//                opponent_rot.e().row(2).transpose(), /// opponent player orientation 3
//                opponent_cage2base_pos_xy_.norm(),
//                40.0, /// opponent cage2base xy position 1
//
//                3.0; /// cage radius 1
  }

  inline void recordReward(Reward *rewards) {
    double poseError = (gc_.head(3) - globalCommandPoint.head(3)).squaredNorm() / (gc_init_.head(3) - globalCommandPoint.head(3)).squaredNorm();

    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d targetVector = rot.e().transpose() * (globalCommandPoint.head(3) - gc_.head(3)) / (globalCommandPoint.head(3) - gc_.head(3)).norm();

    rewForwardVel_ = exp(-(gv_.head(2) / (gv_.head(2).norm() + 1e-5)  - targetVector.head(2)).squaredNorm() / 0.25); // std::min(0.5, (gv_.head(2) - targetVector.head(2)*0.5).norm());
    rewMove2Opponent_ = exp(-(gc_.head(3) - globalCommandPoint.head(3)).squaredNorm()/0.25);
//    if((gc_.head(2) - globalCommandPoint.head(2)).norm() < 0.2 ) rewMove2Opponent_ = 0.0;
//    rewTorque_ = anymal_->getGeneralizedForce().squaredNorm();
    rewBaseMotion_ = (0.8 * bodyLinearVel_[2] * bodyLinearVel_[2] + 0.4* fabs(bodyAngularVel_[0]) + 0.4 * fabs(bodyAngularVel_[1]));
    rewStayMotion_ = std::min(100, continuousGoalCount);
    rewJointPosition = (gc_.tail(nJoints_) - gc_init_.tail(nJoints_)).norm();
//    rewBaseHeight = pow((gc_[2] - gc_init_[2]),2);

    //opponent_gc_init_.head(2) << opponent_cage2base_pos_xy_;

    rewards->record("forwardVel", rewForwardVel_);
    rewards->record("move2Opponent",rewMove2Opponent_);
    rewards->record("stayMotion",rewStayMotion_);
//    rewards->record("torque", rewTorque_);
//    rewards->record("takeGoodPose", rewTakeGoodPose);
//    rewards->record("opponent2CageDist", rewOpponent2CageDist_);
//    rewards->record("pushOpponentOff", rewPushOpponentOff_);
    rewards->record("baseMotion", rewBaseMotion_);
    rewards->record("jointPosition", rewJointPosition);
//    rewards->record("baseHeight", rewBaseHeight);
    rewards->record("curriculumLevel", curriculumLevel);
    rewards->record("commandSuccessCount", commandSuccessCount);
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

  void setCageRadius(const double &cage_radius) {
    cage_radius_ = cage_radius;
  }

  void setEpisodeTime(const double &episodeTime){
    episodeTime_ = episodeTime;
  }

  void updateCurrentTime(const double &controlTime){
    currentTime_ += controlTime;
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

  bool checkTerminateEpisode()
  {
    if(currentTime_ > episodeTime_) return true;
    else return false;
  }

  void randomizeOpponentPosition(raisim::World *world,  double theta){
    auto oppositeAngle = uniDistCage_(gen_) * M_PI;
    double radius = 1.5;
//    opponent_->setPosition(0,0,0.5);`
    if(isOpponentPosCurriculum){
      double prob = uniDist_(gen_);

      if(prob > 0.5) radius = 1.0 + std::min(0.5, (cage_radius_ / 2) * std::min(1.0, ((double)iter_ / cageRadiusCurriculumIter)));
      else radius = 1.0 + uniDist_(gen_) * std::min(0.5, (cage_radius_ / 2) * std::min(1.0, ((double)iter_ / cageRadiusCurriculumIter)));
    }
    else radius = 1.5;

    opponent_gc_init_.head(3) << radius * std::cos(theta + oppositeAngle), radius * std::sin(theta + oppositeAngle), 0.5;//1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
    opponent_gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
//    opponent_->setPosition(radius * std::cos(theta + oppositeAngle), radius * std::sin(theta + oppositeAngle), 0.38);
//    opponent_->setOrientation(cos(theta / 2), 0, 0, sin(theta / 2));
//    opponent_gc_init_.head(3) << radius * std::cos(theta + oppositeAngle), radius * std::sin(theta + oppositeAngle), 0.38;//1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
  }

  void opponentMassCurriculum(){
    if(curriculumLevel > opponentMassCurriculumStartLevel) {
      opponent_->setMass(0, 2.0 + 16.8 * std::min(1.0, ((double) (curriculumLevel - opponentMassCurriculumStartLevel) / (double) (maxcurriculumLevel - opponentMassCurriculumEndLevel))));
      opponent_->updateMassInfo();
    }
  }

  void opponentBaseCollisionCurriculum(){
    opponent_base_collision_size_(0) = 1.0 - (1.0 - 0.537) * std::min(1.0, ((double)(curriculumLevel - opponentBaseCollisionCurriculumStartLevel) / (double)(opponentBaseCollisionCurriculumEndLevel)));
    opponent_base_collision_size_(1) = 0.5 - (0.5 - 0.27) * std::min(1.0, ((double)(curriculumLevel - opponentBaseCollisionCurriculumStartLevel) / (double)(opponentBaseCollisionCurriculumEndLevel)));
    opponent_base_collision_size_(2) = 0.4 - (0.4 - 0.24) * std::min(1.0, ((double)(curriculumLevel - opponentBaseCollisionCurriculumStartLevel) / (double)(opponentBaseCollisionCurriculumEndLevel)));
    opponent_->setCollisionObjectShapeParameters(0, opponent_base_collision_size_);
  }

  void curriculumUpdate(int iter){
    iter_ = iter;
  }

  void curriculumLevelUpdate() {

    if (commandSuccessCount > 0) {
      curriculumLevel += 1;
    } else if (maxContinuousGoalCount < 100) {
      curriculumLevel -= 1;
    }

    if (curriculumLevel < 0) {
      curriculumLevel = 0;
    } else if (curriculumLevel > maxcurriculumLevel) {
      curriculumLevel = uniDistInt_(gen_);
    }
  }

  void randomizeBoxVelocity(raisim::World *world){
      anymal_->getState(gc_, gv_);


      Eigen::Vector3d poseError = (gc_.head(3)) / ((gc_.head(3)).norm() + 1e-5);
      raisim::Vec<3> boxForce;
      boxForce[0] = poseError(0);
      boxForce[1] = poseError(1);
      boxForce[2] = 0.0;

      raisim::Vec<3> auxForce;


      raisim::Vec<3> shearForce;
      shearForce[0] = -poseError(1);
      shearForce[1] = poseError(0);
      shearForce[2] = 0.0;

      shearForce = shearForce / (shearForce.norm() + 1e-5);


      if(currentTime_ < 1e-5){
        externalForceCount=0;
        shearForce *= uniDist_(gen_) > 0.5 ? 1.0 : -1.0;

        auxForce[0] = uniDistBothSide_(gen_);
        auxForce[1] = uniDistBothSide_(gen_);
        auxForce[2] = -1 * uniDist_(gen_);
        auxForce = auxForce / (auxForce.norm() + 1e-5);
        externalForceCount++;
      }
      else if(currentTime_ > 3.0 && externalForceCount == 1){
        shearForce *= uniDist_(gen_) > 0.5 ? 1.0 : -1.0;

        auxForce[0] = uniDistBothSide_(gen_);
        auxForce[1] = uniDistBothSide_(gen_);
        auxForce[2] = -1 * uniDist_(gen_);
        auxForce = auxForce / (auxForce.norm() + 1e-5);
        externalForceCount++;
      }
      else if(currentTime_ > 8.0 && externalForceCount == 2){
        shearForce *= uniDist_(gen_) > 0.5 ? 1.0 : -1.0;

        auxForce[0] = uniDistBothSide_(gen_);
        auxForce[1] = uniDistBothSide_(gen_);
        auxForce[2] = -1 * uniDist_(gen_);
        auxForce = auxForce / (auxForce.norm() + 1e-5);
        externalForceCount++;}


      double randRadialForce = uniDistBothSide_(gen_);
      boxForce = (boxForce * (randRadialForce * 0.4 + 1.5) + auxForce * 0.8 + shearForce);
      boxForce = boxForce / (boxForce.norm() + 1e-5);


      if(curriculumLevel > 100){
        boxExternalForce_ = boxForce.e() * 4.0 * ((double)(curriculumLevel) -100.) / 100. * 30.0;
      }
  }

  void commandPointUpdate(){
    if(isCommandPointCurriculum) commandPointCurriculum();
    else{
      if(changeGoal) {
        commandPointCount++;
        globalCommandPoint(0) = uniDistBothSide_(gen_);
        globalCommandPoint(1) = uniDistBothSide_(gen_);
        globalCommandPoint(2) = 0.0;
        globalCommandPoint = uniDist_(gen_) * cage_radius_ * 0.99 * globalCommandPoint / (globalCommandPoint.norm() + 1e-5);
        globalCommandPoint(2) = 0.5;
        changeGoal = false;
      }
      else{
        checkcommandPointSuccess();
//        if(currentTime_ > 3.0){
//          if(curriculumLevel < 100) {
//            if (checkcommandPointSuccess(true) == 1) changeGoal = true;
//          }
//          else{
//          if (checkcommandPointSuccess(true) == 1 && commandPointCount == 2) changeGoal = true;
//          }
//        }
      }
    }
  }

  void commandPointCurriculum(){
      Eigen::Vector3d tempCommandPoint;
      tempCommandPoint.setZero();
      anymal_->getState(gc_, gv_);
      double prob = uniDist_(gen_);
      double rand_radius;

      if(currentTime_ < 1e-5 || changeGoal) {
//        commandPointCount = 0;
//        commandPointCount++;
        while (true) {
          tempCommandPoint(0) = uniDistBothSide_(gen_);
          tempCommandPoint(1) = uniDistBothSide_(gen_);
          tempCommandPoint(2) = 0.0;
          double level = std::min(1.0, ((double)(curriculumLevel) / 100.));
          if(level < 0.1) level = 0.1;
          if(curriculumLevel < 100) {
            rand_radius = uniDist_(gen_) * level * cage_radius_;
            tempCommandPoint(0) = uniDistBothSide_(gen_) * rand_radius * 0.9;
            tempCommandPoint(1) = sqrt(pow(rand_radius * 0.9, 2) - tempCommandPoint(0) * tempCommandPoint(0));
            tempCommandPoint(2) = 0.5;
          }
          else{
            rand_radius = uniDist_(gen_) * cage_radius_;
            tempCommandPoint(0) = uniDistBothSide_(gen_) * rand_radius * 0.9;
            tempCommandPoint(1) = sqrt(pow(rand_radius * 0.95, 2) - tempCommandPoint(0) * tempCommandPoint(0));
            tempCommandPoint(2) = 0.5;
          }
//          else tempCommandPoint = tempCommandPoint / (tempCommandPoint.norm() + 1e-5) * level * cage_radius_ * 0.8 * std::min(0.2, uniDist_(gen_));
          if ((tempCommandPoint.head(2)).norm() < cage_radius_) break;
          std::cout << "co" << tempCommandPoint.transpose() << "//" << gc_.head(2).transpose() << std::endl;
        }
//        if(prob < 0.1){
//          globalCommandPoint = gc_.head(3);
//          globalCommandPoint(2) = 0.5;
//        }
//        else {
        globalCommandPoint = tempCommandPoint;
        globalCommandPoint(2) = 0.5;
//        }
        changeGoal = false;
      }
      checkcommandPointSuccess();
      if(currentTime_ > 5.0 && checkcommandPointSuccess(true) == 1){
        changeGoal = true;
      }
//      else if(currentTime_ > 5.0 && commandPointCount == 1){
//
//        commandPointCount++;
//        while (true) {
//          tempCommandPoint(0) = uniDistBothSide_(gen_);
//          tempCommandPoint(1) = uniDistBothSide_(gen_);
//          tempCommandPoint(2) = 0.0;
//          double level = std::min(1.0, ((double)(curriculumLevel) / 100.));
//          if(level < 0.1) level = 0.1;
//          if(curriculumLevel < 100) {
//            rand_radius = uniDist_(gen_) * level * cage_radius_;
//            tempCommandPoint(0) = uniDistBothSide_(gen_) * rand_radius * 0.9;
//            tempCommandPoint(1) = sqrt(pow(rand_radius * 0.9, 2) - tempCommandPoint(0) * tempCommandPoint(0));
//            tempCommandPoint(2) = 0.5;
//          }
//          else{
//            rand_radius = uniDist_(gen_) * cage_radius_;
//            tempCommandPoint(0) = uniDistBothSide_(gen_) * rand_radius * 0.9;
//            tempCommandPoint(1) = sqrt(pow(rand_radius * 0.95, 2) - tempCommandPoint(0) * tempCommandPoint(0));
//            tempCommandPoint(2) = 0.5;
//          }
////          else tempCommandPoint = tempCommandPoint / (tempCommandPoint.norm() + 1e-5) * level * cage_radius_ * 0.8 * std::min(0.2, uniDist_(gen_));
//          if ((tempCommandPoint.head(2)).norm() < cage_radius_) break;
//          std::cout << "co" << tempCommandPoint.transpose() << "//" << gc_.head(2).transpose() << std::endl;
//        }
//        if(prob < 0.1){
//          globalCommandPoint = gc_.head(3);
//          globalCommandPoint(2) = 0.5;
//        }
//        else {
//          globalCommandPoint = tempCommandPoint;
//          globalCommandPoint(2) = 0.5;
//        }
//      }
  }

  int checkcommandPointSuccess(bool isvalidation=false){
    if((gc_.head(3) - globalCommandPoint.head(3)).norm() < 0.2) {
      continuousGoalCount++;
      if(maxContinuousGoalCount < continuousGoalCount) maxContinuousGoalCount = continuousGoalCount;
    }
    else{
      if(maxContinuousGoalCount < continuousGoalCount) maxContinuousGoalCount = continuousGoalCount;
      continuousGoalCount = 0;
    }
    if(isvalidation && maxContinuousGoalCount > 200){
      commandSuccessCount++;
      if(commandSuccessCount > 2) commandSuccessCount = 2;
      return 1;
    }
    return 0;

  }

  raisim::ArticulatedSystem *anymal_, *opponent_;

  //RandomizeRelated
  bool israndomizeOpponentPosition = true;
  bool israndomizeOpponentVelocity = true;
  bool isOpponentPosCurriculum = false;
  bool isOpponentMassCurriculum= false;
  bool isOpponentBaseCollisionCurriculum= false;
  bool isCageRadiusCurriculum = false;
  bool isCommandPointCurriculum = false;

  double cageRadiusCurriculumIter = 1000.;
  double boxExternelForceCurriculumIter = 2000.;
  double boxMassCurriculumIter = 3000.;
  int curriculumLevel = 0.;
  int maxcurriculumLevel = 200;
  int opponentMassCurriculumStartLevel = 100;
  int opponentMassCurriculumEndLevel = 200;
  int opponentBaseCollisionCurriculumStartLevel = 0;
  int opponentBaseCollisionCurriculumEndLevel = 100;

  Eigen::Vector3d boxExternalForce_;
  Eigen::Vector3d globalCommandPoint, bodyCommandPoint;
  int commandPointCount = 0;
  int commandSuccessCount = 0;
  int continuousGoalCount = 0;
  int maxContinuousGoalCount = 0;
  bool changeGoal = false;

 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;

  raisim::Visuals *cage_;
  double cage_radius_ = 3.0;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd cage2base_pos_xy_, cage2base_pos_body_;
  Eigen::VectorXd previousAction_, prepreviousAction_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, obDoubleTeacher_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int obDim_ = 0, realObDim_ = 0, obDimTeacher_ = 74, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;



  //opponent robot's data
  Eigen::VectorXd opponent_gc_, opponent_gv_, opponent_gc_init_, opponent_gv_init_, opponent_pTarget_;
  Eigen::Vector3d opponent_bodyLinearVel_, opponent_bodyAngularVel_;
  Eigen::VectorXd opponent_cage2base_pos_xy_;
  raisim::Vec<3> opponent_base_collision_size_;


  int externalForceCount = 0;

  //Train Related
  int iter_ = 0;
  double episodeTime_ = 10.0;
  double controlTime_ = 0.01;
  double currentTime_ = 0.0;

  //RewardRelated
  double rewForwardVel_ = 0., rewMove2Opponent_ = 0., rewTorque_ = 0., rewTakeGoodPose = 0., rewOpponent2CageDist_ = 0., rewPushOpponentOff_ = 0., rewBaseMotion_=0.,
          rewJointPosition = 0., rewBaseHeight=0., rewStayMotion_ = 0.;

  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::uniform_real_distribution<double> uniDistBothSide_;
  thread_local static std::uniform_real_distribution<double> uniDistCage_;
  thread_local static std::uniform_int_distribution<int> uniDistInt_;
  thread_local static std::mt19937 gen_;

};
thread_local std::mt19937 raisim::PretrainingAnymalController_20233319::gen_;
thread_local std::uniform_real_distribution<double> raisim::PretrainingAnymalController_20233319::uniDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::PretrainingAnymalController_20233319::uniDistBothSide_(-1., 1.);
thread_local std::uniform_real_distribution<double> raisim::PretrainingAnymalController_20233319::uniDistCage_(0.8, 1.2);
thread_local std::uniform_int_distribution<int> raisim::PretrainingAnymalController_20233319::uniDistInt_(80, 200);
}


