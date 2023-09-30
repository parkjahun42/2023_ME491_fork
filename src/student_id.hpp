#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

class TicTacToe {

 public:
  std::vector<Eigen::MatrixXd> terminateStates;
    enum class Type{
    KEEP_GO = -1,
    LOSE, //0
    DRAW, //1
    WIN //2
  }type_=Type::KEEP_GO;

  TicTacToe(int mapsize);
  void train();
  void printCurrentState(const Eigen::MatrixXd &map);
  void setState(const Eigen::MatrixXd &map);
  void setValue(const Eigen::MatrixXd &map, double value);
  double getValue(const Eigen::MatrixXd &map);
  TicTacToe::Type checkTermination(Eigen::MatrixXd map);

  //Become private



 private:
  void initTerminateStates();
  Eigen::VectorXi checkEmptyState(const Eigen::MatrixXd &map);
  double takeActions(const Eigen::MatrixXd &map);
  double takeActionsOppenentFirst(const Eigen::MatrixXd &map);
  double takeOpponentActionAndGetRewardValue(const Eigen::MatrixXd &actionMap);


  Eigen::MatrixXd map_;
  int mapTotalCount_, mapSize_;
  double gamma_ = 0.98;
  double valueTable[3][3][3][3][3][3][3][3][3] = {0.0};

};

TicTacToe::TicTacToe(int mapsize) {
  mapSize_ = mapsize;
  mapTotalCount_ = mapsize*mapsize;

  map_.setZero(mapsize, mapsize);
  initTerminateStates();
}

void TicTacToe::train(){
  Eigen::MatrixXd map;
  map.setZero(mapSize_, mapSize_);
  takeActions(map);
  takeActionsOppenentFirst(map);
}

void TicTacToe::printCurrentState(const Eigen::MatrixXd &map) {
  std::cout << "\n==\t==\t==" << std::endl;
  std::cout <<  map(0) << "\t" << map(1) << "\t" << map(2) << std::endl;
  std::cout <<  map(3) << "\t" << map(4) << "\t" << map(5) << std::endl;
  std::cout <<  map(6) << "\t" << map(7) << "\t" << map(8) << std::endl;
  std::cout << "==\t==\t==" << std::endl;
}

void TicTacToe::setState(const Eigen::MatrixXd &map) {
  map_ = map;
}

void TicTacToe::setValue(const Eigen::MatrixXd &map, double value)
{
  valueTable[(int)map(0)+1][(int)map(1)+1][(int)map(2)+1][(int)map(3)+1][(int)map(4)+1][(int)map(5)+1][(int)map(6)+1][(int)map(7)+1][(int)map(8)+1] = value;
}

double TicTacToe::getValue(const Eigen::MatrixXd &map) {
  return valueTable[(int)map(0)+1][(int)map(1)+1][(int)map(2)+1][(int)map(3)+1][(int)map(4)+1][(int)map(5)+1][(int)map(6)+1][(int)map(7)+1][(int)map(8)+1];
}

//Should be changed when mapsize become different
void TicTacToe::initTerminateStates() {
  Eigen::Matrix3d terminateState;

  terminateState << 1, 1, 1, 0, 0, 0, 0, 0, 0;
  terminateStates.push_back(terminateState);
  terminateState << 0, 0, 0, 1, 1, 1, 0, 0, 0;
  terminateStates.push_back(terminateState);
  terminateState << 0, 0, 0, 0, 0, 0, 1, 1, 1;
  terminateStates.push_back(terminateState);
  terminateState << 1, 0, 0, 1, 0, 0, 1, 0, 0;
  terminateStates.push_back(terminateState);
  terminateState << 0, 1, 0, 0, 1, 0, 0, 1, 0;
  terminateStates.push_back(terminateState);
  terminateState << 0, 0, 1, 0, 0, 1, 0, 0, 1;
  terminateStates.push_back(terminateState);
  terminateState << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  terminateStates.push_back(terminateState);
  terminateState << 0, 0, 1, 0, 1, 0, 1, 0, 0;
  terminateStates.push_back(terminateState);
}

TicTacToe::Type TicTacToe::checkTermination(Eigen::MatrixXd map) {

  for(auto terminateState:terminateStates){
    int markSum = (map.array() * terminateState.array()).sum();
    if(markSum == mapSize_){
//      std::cout << "User Wins!" << std::endl;
      return TicTacToe::Type::WIN;
    }
    else if(markSum == -mapSize_){
//      std::cout << "Oppenent Wins!" << std::endl;
      return TicTacToe::Type::LOSE;
    }
  }

  int turnCount = 0;

  for(int i = 0 ; i<mapTotalCount_; i++)
  {
    if(map(i)!=0) turnCount++;
  }

  if(turnCount==mapTotalCount_)
  {
//    std::cout << "Draw!!" << std::endl;
    return TicTacToe::Type::DRAW;
  }
  else
  {
//    std::cout << "Keep going!!" << std::endl;
    return TicTacToe::Type::KEEP_GO;
  }
}

Eigen::VectorXi TicTacToe::checkEmptyState(const Eigen::MatrixXd &map) {
  int emptyCounter = 0;
  Eigen::VectorXi mapEmptyIndex;
  mapEmptyIndex.setZero(mapSize_*mapSize_ + 1);


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

double TicTacToe::takeActions(const Eigen::MatrixXd &map) {
    Eigen::VectorXi mapEmptyIndex;
    TicTacToe::Type nextStateType;
    Eigen::VectorXd tempValue;
    Eigen::MatrixXd actionMap;
    mapEmptyIndex = checkEmptyState(map);
    tempValue.setZero(mapEmptyIndex(mapTotalCount_));
    actionMap.setZero(mapSize_, mapSize_);
    nextStateType = checkTermination(map);
    if (nextStateType != TicTacToe::Type::KEEP_GO) {  //Check Termination and give Value
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

double TicTacToe::takeActionsOppenentFirst(const Eigen::MatrixXd &map){ //When oppenent start first
  Eigen::VectorXi mapEmptyIndex;
  TicTacToe::Type nextStateType;
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

      setValue(map, maxValue);
  }




  return 1.0;
}

double TicTacToe::takeOpponentActionAndGetRewardValue(const Eigen::MatrixXd &actionMap) {
  TicTacToe::Type nextStateType;
  Eigen::Matrix<int, 10, 1> mapEmptyIndex;
  Eigen::Matrix<double, 3, 3> nextMap;
  double reward = 0.0, value = 0.0;

  nextStateType = checkTermination(actionMap);
  if(nextStateType != TicTacToe::Type::KEEP_GO){  //Terminate when user finished game. give value because we should give value for next state and multiply discount factor
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
double getOptimalValue(Eigen::Matrix3d state){
  /// TODO
  double value;
  TicTacToe tictactoe(3);

  //Train
  tictactoe.train();

  return tictactoe.getValue(state); // return optimal value
}

