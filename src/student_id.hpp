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

  DotsAndBoxes(int mapsize);
  void train();
  void printCurrentState(const Eigen::VectorXd &map);
  void setState(const Eigen::VectorXd &map);
  void setValue(const Eigen::VectorXd &map, double value);
  double getValue(const Eigen::VectorXd &map);
  DotsAndBoxes::Type checkTermination(Eigen::VectorXd map);

  //Become private



 private:
  void initTerminateStates();
  void initRewardStates();
  void initPolicyTable();
  double giveReward(Eigen::VectorXd &map, Eigen::VectorXd &preMap);
  Eigen::VectorXi checkEmptyState(const Eigen::VectorXd &map);
  double takeActions(const Eigen::MatrixXd &map);
  double takeActionsOppenentFirst(const Eigen::MatrixXd &map);
  double takeOpponentActionAndGetRewardValue(const Eigen::MatrixXd &actionMap);


  Eigen::VectorXd map_;
  int mapTotalCount_, mapSize_;
  double gamma_ = 0.98;
  double valueTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0.0};
  int policyTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0};

};

DotsAndBoxes::DotsAndBoxes(int mapsize) {
  mapSize_ = mapsize;
  mapTotalCount_ = mapSize_;

  map_.setZero(mapsize);
  initTerminateStates();
}

void DotsAndBoxes::train(){
  Eigen::VectorXd map;
  map.setZero(mapSize_);
  takeActions(map);
  takeActionsOppenentFirst(map);
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
  if(map(6) == 1) std::cout << "|  " ;
  else std::cout << "   " ;
  if(map(8) == 1) std::cout << "|  " ;
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
  if(map(7) == 1) std::cout << "|  " ;
  else std::cout << "   " ;
  if(map(9) == 1) std::cout << "|  " ;
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

void DotsAndBoxes::setValue(const Eigen::VectorXd map, double value)
{
  valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = value;
}

double DotsAndBoxes::getValue(const Eigen::VectorXd &map) {
  return valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
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
  rewardState << 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0;
  rewardState << 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0;
  rewardState << 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1;
  rewardStates.push_back(rewardState);
}

DotsAndBoxes::Type DotsAndBoxes::checkTermination(Eigen::VectorXd map) {

  for(auto terminateState:terminateStates){
    int markSum = (map.array() * terminateState.array()).sum();
    if(markSum == mapSize_){
      return DotsAndBoxes::Type::TERMINATE;
    }
    else
      return DotsAndBoxes::Type::KEEP_GO;
    }
}


double DotsAndBoxes::giveReward(Eigen::VectorXd &map, Eigen::VectorXd &preMap) {

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

double DotsAndBoxes::policyEvaluation(const Eigen::VectorXd &map){
  //Fixed policy, get Value function
  Eigen::VectorXi mapEmptyIndex;
  DotsAndBoxes::Type nextStateType;
  Eigen::VectorXd tempValue;
  Eigen::VectorXd actionMap;
  mapEmptyIndex = checkEmptyState(map);
  tempValue.setZero(mapEmptyIndex(mapTotalCount_));
  actionMap.setZero(mapSize_);
  nextStateType = checkTermination(map);
  if (nextStateType != DotsAndBoxes::Type::KEEP_GO) {  //Check Termination and give Value
      setValue(map, (double)(nextStateType) / 2);
      return (double) nextStateType / 2;
  }
  else {

      actionMap = map;
      actionMap(getActionFromPolicy(map)) = 1.0;
      tempValue(0) = takeOpponentActionAndGetRewardValue(
                actionMap);

      setValue(map, tempValue(0));

      return tempValue(0);
  }
}

double DotsAndBoxes::takeActions(const Eigen::VectorXd &map) {
    Eigen::VectorXi mapEmptyIndex;
    DotsAndBoxes::Type nextStateType;
    Eigen::VectorXd tempValue;
    Eigen::VectorXd actionMap;
    mapEmptyIndex = checkEmptyState(map);
    tempValue.setZero(mapEmptyIndex(mapTotalCount_));
    actionMap.setZero(mapSize_);
    nextStateType = checkTermination(map);
    if (nextStateType != DotsAndBoxes::Type::KEEP_GO) {  //Check Termination and give Value
        setValue(map, (double)(nextStateType) / 2);
        return (double) nextStateType / 2;
    }
    else {
        for (int i = mapEmptyIndex(mapTotalCount_) - 1; i >= 0; i--) { //Iterate actions for all empty grid
            actionMap = map;
            actionMap(mapEmptyIndex(i)) = 1.0;
//    printCurrentState(actionMap);
            tempValue(i) = takeOpponentActionAndGetRewardValue(
                    actionMap); //Get Value(Reward) for action
        }

        double maxValue;
        for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) {  //Take max value from all actions
            if (i == 0) maxValue = tempValue(0);
            if (tempValue(i) > maxValue) maxValue = tempValue(i);
        }

        setValue(map, maxValue);

        return maxValue;
    }
}

double DotsAndBoxes::takeActionsOppenentFirst(const Eigen::MatrixXd &map){ //When oppenent start first
  Eigen::VectorXi mapEmptyIndex;
  DotsAndBoxes::Type nextStateType;
  Eigen::VectorXd tempValue;
  Eigen::MatrixXd actionMap, opponentMap;

  actionMap.setZero(mapSize_, mapSize_);
  opponentMap.setZero(mapSize_, mapSize_);
  for(int j = 0; j<9; j++)
  {
    opponentMap = map;
    opponentMap(j) = -1.0;
    mapEmptyIndex = checkEmptyState(opponentMap);
    tempValue.setZero(mapEmptyIndex(mapTotalCount_));
    for(int i =mapEmptyIndex(mapTotalCount_)-1; i>= 0; i--)
    {
      actionMap = opponentMap;
      actionMap(mapEmptyIndex(i)) = 1.0;
//      printCurrentState(actionMap);
      tempValue(i) = takeOpponentActionAndGetRewardValue(actionMap);
    }
      double maxValue;
      for(int i =0; i<mapEmptyIndex(mapTotalCount_); i++)
      {
        if(i == 0) maxValue = tempValue(0);
        if(tempValue(i) > maxValue) maxValue = tempValue(i);
      }

      setValue(opponentMap, maxValue);
  }




  return 1.0;
}

double DotsAndBoxes::takeOpponentActionAndGetRewardValue(const Eigen::MatrixXd &actionMap) {
  DotsAndBoxes::Type nextStateType;
  Eigen::Matrix<int, 10, 1> mapEmptyIndex;
  Eigen::Matrix<double, 3, 3> nextMap;
  double reward = 0.0, value = 0.0;

  nextStateType = checkTermination(actionMap);
  if(nextStateType != DotsAndBoxes::Type::KEEP_GO){  //Terminate when user finished game. give value because we should give value for next state and multiply discount factor
//    reward = (double)nextStateType / 2;
    value = (double)nextStateType / 2; //Terminate State
  }
  else {
    reward = 0.0;
    mapEmptyIndex = checkEmptyState(actionMap);

    for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) { //caculate value for all opponent action
      nextMap = actionMap;
      nextMap(mapEmptyIndex(i)) = -1.0;
      value += takeActions(nextMap) / mapEmptyIndex(mapTotalCount_);
    }
  }

  return reward + gamma_ * value;
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
double getOptimalValue(const Eigen::Vector<int, 12>& state){
  // return the optimal value given the state
  /// TODO


  return 0.0;  // return optimal value
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
int getOptimalAction(const Eigen::Vector<int, 12>& state){
  // return one of the optimal actions given the state.
  // the action should be represented as a state index, at which a line will be drawn.
  /// TODO

  return 0;  // return optimal action
}