#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

class DotsAndBoxes {

 public:
  std::vector<Eigen::VectorXd> terminateStates;
  std::vector<Eigen::VectorXd> rewardStates;
    enum class Type{
    KEEP_GO = -1,
    TERMINATE, //0
//    DRAW, //1
//    WIN //2
  }type_=Type::KEEP_GO;

    enum class TakeActionType{
      EVALUATION = 0,
      IMPROVEMENT,
    }takeActionType_ = TakeActionType::EVALUATION;

  DotsAndBoxes(int mapsize);
  void train();
  void printCurrentState(const Eigen::VectorXd &map);
  void setState(const Eigen::VectorXd &map);
  void setValue(const Eigen::VectorXd &map, double value);
  void setPolicy(const Eigen::VectorXd &map, int policy);
  double getValue(const Eigen::VectorXd &map);
  int getPolicy(const Eigen::VectorXd &map);
  DotsAndBoxes::Type checkTermination(const Eigen::VectorXd map);

  //Become private



 private:
  int count_=0;
  void initTerminateStates();
  void initRewardStates();
  void initPolicyTable();
  double giveReward(const Eigen::VectorXd &map, const  Eigen::VectorXd &preMap);
  Eigen::VectorXi checkEmptyState(const Eigen::VectorXd &map);
  int getActionFromPolicy(const Eigen::VectorXd &map, bool isFirst);
  double takeActions(const Eigen::VectorXd &map, TakeActionType takeActionType, bool isFirst);
  double takeActionsOppenentFirst(const Eigen::VectorXd &map);
  double takeOpponentActionAndGetRewardValue(const Eigen::VectorXd &actionMap, TakeActionType takeActionType, bool isFirst);


  Eigen::VectorXd map_;
  int mapTotalCount_, mapSize_;
  double gamma_ = 1.0;
  double valueTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0.0};
  int policyTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0};

};

DotsAndBoxes::DotsAndBoxes(int mapsize) {
  mapSize_ = mapsize;
  mapTotalCount_ = mapSize_;

  map_.setZero(mapsize);
  initTerminateStates();
  initRewardStates();
}

void DotsAndBoxes::train(){
  Eigen::VectorXd map;
  map.setZero(mapSize_);
  takeActions(map, TakeActionType::EVALUATION, true);
  map.setZero(mapSize_);
  takeActions(map, TakeActionType::IMPROVEMENT, true);
//  takeActionsOppenentFirst(map);
}

void DotsAndBoxes::printCurrentState(const Eigen::VectorXd &map) {
  std::cout << "\n==\t==\t==" << std::endl;
  std::cout << "o";
  if(map(0) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o";
  if(map(1) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o" << std::endl;
  if(map(6) == 1) std::cout << "|    " ;
  else std::cout << "   " ;
  if(map(8) == 1) std::cout << "|   " ;
  else std::cout << "   " ;
  if(map(10) == 1) std::cout << "|" << std::endl;
  else std::cout << " " << std::endl;
  std::cout << "o";
  if(map(2) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o";
  if(map(3) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o" << std::endl;
  if(map(7) == 1) std::cout << "|    " ;
  else std::cout << "   " ;
  if(map(9) == 1) std::cout << "|    " ;
  else std::cout << "   " ;
  if(map(11) == 1) std::cout << "|" << std::endl;
  else std::cout << " " << std::endl;
  std::cout << "o";
  if(map(4) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o";
  if(map(5) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o" << std::endl;
  std::cout << "==\t==\t==" << std::endl;
}

void DotsAndBoxes::setState(const Eigen::VectorXd &map) {
  map_ = map;
}

void DotsAndBoxes::setValue(const Eigen::VectorXd &map, double value)
{
  valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = value;
}

void DotsAndBoxes::setPolicy(const Eigen::VectorXd &map, int policy)
{
  policyTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = policy;
}

double DotsAndBoxes::getValue(const Eigen::VectorXd &map) {
  return valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
}

int DotsAndBoxes::getPolicy(const Eigen::VectorXd &map)
{
  return policyTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
}

//Should be changed when mapsize become different
void DotsAndBoxes::initTerminateStates() {
  Eigen::VectorXd terminateState;
  terminateState.setZero(mapTotalCount_);

  terminateState << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  terminateStates.push_back(terminateState);
}

void DotsAndBoxes::initRewardStates() {
  Eigen::VectorXd rewardState;
  rewardState.setZero(mapTotalCount_);
  //             0  1  2  3  4  5  6  7  8  9  10  11
  rewardState << 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0;
  rewardStates.push_back(rewardState);
  rewardState << 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0;
  rewardStates.push_back(rewardState);
  rewardState << 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0;
  rewardStates.push_back(rewardState);
  rewardState << 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1;
  rewardStates.push_back(rewardState);
}

DotsAndBoxes::Type DotsAndBoxes::checkTermination(const Eigen::VectorXd map) {

  for(auto terminateState:terminateStates){
    int markSum = (map.array() * terminateState.array()).sum();
    if(markSum == mapSize_){
      return DotsAndBoxes::Type::TERMINATE;
    }
    else
      return DotsAndBoxes::Type::KEEP_GO;
    }
}


double DotsAndBoxes::giveReward(const Eigen::VectorXd &map, const Eigen::VectorXd &preMap) {

  double rewardSum = 0.0;

  for (auto rewardState : rewardStates) {
    int markSum = (map.array() * rewardState.array()).sum();
    int preMarkSum = (preMap.array() * rewardState.array()).sum();
    if (markSum == 4 && preMarkSum == 3) {
      rewardSum++;
    }
  }

  return rewardSum;
}

Eigen::VectorXi DotsAndBoxes::checkEmptyState(const Eigen::VectorXd &map) {
  int emptyCounter = 0;
  Eigen::VectorXi mapEmptyIndex;
  mapEmptyIndex.setZero(mapSize_ + 1);


  for(int i = 0; i<mapTotalCount_; i++)
  {
    if(map(i) == 0){
      mapEmptyIndex(emptyCounter) = i;
      emptyCounter++;
    }
  }
  mapEmptyIndex(mapTotalCount_) = emptyCounter;

  return mapEmptyIndex;

}

int DotsAndBoxes::getActionFromPolicy(const Eigen::VectorXd &map, bool isFirst){
  if(checkTermination(map) !=DotsAndBoxes::Type::KEEP_GO){
    std::cout << "Terminate state! Cant get action!" << std::endl;
    return -1;
  }
  if(isFirst){
    Eigen::VectorXi mapEmptyIndex;
    mapEmptyIndex = checkEmptyState(map);
    setPolicy(map, mapEmptyIndex[0]);
    return mapEmptyIndex[0];
  }
  else{
    return getPolicy(map);
  }
}

//이슈 : 액션을 취한 다음에 넘길때 내 차롄지, 상대방 차례인지를 알아야 함.
// 또 내 차례면 바로 takeActions 넣어줘 value를 내뱉으면 됨.
// 상대방 차례면 랜덤으로 상대방 action 넣은 state를 takeActions에 넣어줘서 value 내뱉으면 됨.
double DotsAndBoxes::takeActions(const Eigen::VectorXd &map, TakeActionType takeActionType, bool isFirst){

  Eigen::VectorXi mapEmptyIndex;
  DotsAndBoxes::Type nextStateType;
  double nextStateValue = 0.0, reward = 0.0;
  Eigen::VectorXd tempValue;
  Eigen::VectorXd actionMap;
  mapEmptyIndex = checkEmptyState(map);
  tempValue.setZero(mapEmptyIndex(mapTotalCount_));
  actionMap.setZero(mapTotalCount_);
  nextStateType = checkTermination(map);
  count_++;
  std::cout << count_ << std::endl;
  if (nextStateType != DotsAndBoxes::Type::KEEP_GO) {  //Check Termination and give Value
//    printCurrentState(map);
    return 0.0;
  }
  else {
    if (takeActionType == TakeActionType::EVALUATION) { //Fixed policy, get Value function
      actionMap = map;
      actionMap(getActionFromPolicy(map, isFirst)) = 1.0;
      reward = giveReward(actionMap, map); // Caculate Reward
      printCurrentState(actionMap);
      if (reward > 1e-4) {
        nextStateValue = takeActions(actionMap, takeActionType, isFirst);
      }
      else nextStateValue = takeOpponentActionAndGetRewardValue(actionMap, takeActionType, isFirst); //Get v(s') by recursive function

      setValue(map, reward + gamma_ * nextStateValue);

      return reward + gamma_ * nextStateValue;

    }
    else if (takeActionType == TakeActionType::IMPROVEMENT) { //Improve policy with q value
      for (int i = mapEmptyIndex(mapTotalCount_) - 1; i >= 0; i--) { //Iterate actions for all empty grid
        actionMap = map;
        actionMap(mapEmptyIndex(i)) = 1.0; //q function (s, a)
        printCurrentState(actionMap);
        reward = giveReward(actionMap, map); // Caculate Reward

        if (reward > 1e-4) {
          tempValue(i) = reward + gamma_ * takeActions(actionMap, takeActionType, isFirst);
        }
        else {
          tempValue(i) = reward + gamma_ * takeOpponentActionAndGetRewardValue(actionMap, takeActionType, isFirst); //Get v(s') by recursive function
        }

      }

      double maxValue, maxValueIndex;
      for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) {  //Improve policy with maximum state action value
        std::cout << "Value: " << tempValue(i) << std::endl;
        if (i == 0) {
          maxValue = tempValue(0);
          maxValueIndex = 0;
        }
        if (tempValue(i) > maxValue) {
          maxValue = tempValue(i);
          maxValueIndex = mapEmptyIndex(i);
        }
      }

      setPolicy(map, maxValueIndex);

      return getValue(map);
    }
  }



}

//double DotsAndBoxes::takeActions(const Eigen::VectorXd &map) {
//    Eigen::VectorXi mapEmptyIndex;
//    DotsAndBoxes::Type nextStateType;
//    Eigen::VectorXd tempValue;
//    Eigen::VectorXd actionMap;
//    mapEmptyIndex = checkEmptyState(map);
//    tempValue.setZero(mapEmptyIndex(mapTotalCount_));
//    actionMap.setZero(mapSize_);
//    nextStateType = checkTermination(map);
//    if (nextStateType != DotsAndBoxes::Type::KEEP_GO) {  //Check Termination and give Value
//        setValue(map, (double)(nextStateType) / 2);
//        return (double) nextStateType / 2;
//    }
//    else {
//        for (int i = mapEmptyIndex(mapTotalCount_) - 1; i >= 0; i--) { //Iterate actions for all empty grid
//            actionMap = map;
//            actionMap(mapEmptyIndex(i)) = 1.0;
////    printCurrentState(actionMap);
//            tempValue(i) = takeOpponentActionAndGetRewardValue(
//                    actionMap); //Get Value(Reward) for action
//        }
//
//        double maxValue;
//        for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) {  //Take max value from all actions
//            if (i == 0) maxValue = tempValue(0);
//            if (tempValue(i) > maxValue) maxValue = tempValue(i);
//        }
//
//        setValue(map, maxValue);
//
//        return maxValue;
//    }
//}

//double DotsAndBoxes::takeActionsOppenentFirst(const Eigen::VectorXd &map){ //When oppenent start first
//  Eigen::VectorXi mapEmptyIndex;
//  DotsAndBoxes::Type nextStateType;
//  Eigen::VectorXd tempValue;
//  Eigen::MatrixXd actionMap, opponentMap;
//
//  actionMap.setZero(mapSize_, mapSize_);
//  opponentMap.setZero(mapSize_, mapSize_);
//  for(int j = 0; j<9; j++)
//  {
//    opponentMap = map;
//    opponentMap(j) = -1.0;
//    mapEmptyIndex = checkEmptyState(opponentMap);
//    tempValue.setZero(mapEmptyIndex(mapTotalCount_));
//    for(int i =mapEmptyIndex(mapTotalCount_)-1; i>= 0; i--)
//    {
//      actionMap = opponentMap;
//      actionMap(mapEmptyIndex(i)) = 1.0;
////      printCurrentState(actionMap);
//      tempValue(i) = takeOpponentActionAndGetRewardValue(actionMap);
//    }
//      double maxValue;
//      for(int i =0; i<mapEmptyIndex(mapTotalCount_); i++)
//      {
//        if(i == 0) maxValue = tempValue(0);
//        if(tempValue(i) > maxValue) maxValue = tempValue(i);
//      }
//
//      setValue(opponentMap, maxValue);
//  }
//
//
//
//
//  return 1.0;
//}

double DotsAndBoxes::takeOpponentActionAndGetRewardValue(const Eigen::VectorXd &actionMap, TakeActionType takeActionType, bool isFirst) {
  DotsAndBoxes::Type nextStateType;
  Eigen::VectorXi mapEmptyIndex;
  Eigen::VectorXd nextMap;
  nextMap.setZero(mapTotalCount_);
  double value = 0.0, oppenentReward = 0.0;

  mapEmptyIndex = checkEmptyState(actionMap);
  nextStateType = checkTermination(actionMap);

  if(nextStateType != DotsAndBoxes::Type::KEEP_GO){  //Terminate when user finished game. give value because we should give value for next state and multiply discount factor
//    printCurrentState(actionMap);
    return 0.0;
  }
  else {
    mapEmptyIndex = checkEmptyState(actionMap);

    for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) { //caculate value for all opponent action
      nextMap = actionMap;
      nextMap(mapEmptyIndex(i)) = 1.0;
      oppenentReward = giveReward(nextMap, actionMap);

      if(takeActionType == TakeActionType::EVALUATION) {
        if (oppenentReward > 1e-4) value += takeOpponentActionAndGetRewardValue(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_);
        else value += takeActions(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_);
      }
      else{
        if (oppenentReward > 1e-4) value += takeOpponentActionAndGetRewardValue(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_);
        else value += takeActions(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_); //getValue(nextMap) / mapEmptyIndex(mapTotalCount_);//
      }
    }
  }

  return value;
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
double getOptimalValue(const Eigen::Vector<int, 12>& state){
  // return the optimal value given the state
  /// TODO
  DotsAndBoxes dotsAndBoxes(12);

  dotsAndBoxes.train();

  return 0.0;  // return optimal value
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
int getOptimalAction(const Eigen::Vector<int, 12>& state){
  // return one of the optimal actions given the state.
  // the action should be represented as a state index, at which a line will be drawn.
  /// TODO

  return 0;  // return optimal action
}