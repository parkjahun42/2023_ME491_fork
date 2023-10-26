#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

class DotsAndBoxes {

 public:
  std::vector<Eigen::VectorXi> terminateStates;
  std::vector<Eigen::VectorXi> rewardStates;
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
  void policyEvaluation(bool isFirst);
  void policyImprovement(bool isFirst);
  double checkPolicyTerminateCondition();
  void train();
  void printCurrentState(const Eigen::VectorXi &map);
  void setState(const Eigen::VectorXi &map);
  void setValue(const Eigen::VectorXi &map, double value);
  void setPolicy(const Eigen::VectorXi &map, int policy);
  double getValue(const Eigen::VectorXi &map);
  int getPolicy(const Eigen::VectorXi &map);
  void setCaculatedValue(const Eigen::VectorXi &map, int policy);
  int getCaculatedValue(const Eigen::VectorXi &map);
  DotsAndBoxes::Type checkTermination(const Eigen::VectorXi map);

  //Become private



 private:
  int count_=0;
  void initTerminateStates();
  void initRewardStates();
  double giveReward(const Eigen::VectorXi &map, const  Eigen::VectorXi &preMap);
  Eigen::VectorXi checkEmptyState(const Eigen::VectorXi &map);
  int getActionFromPolicy(const Eigen::VectorXi &map, bool isFirst);
  double takeActions(const Eigen::VectorXi &map, TakeActionType takeActionType, bool isFirst);
  double takeActionsOppenentFirst(const Eigen::VectorXi &map);
  double takeOpponentActionAndGetRewardValue(const Eigen::VectorXi &actionMap, TakeActionType takeActionType, bool isFirst);


  Eigen::VectorXi map_;
  int mapTotalCount_, mapSize_;
  double gamma_ = 1.0;
  double valueTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0.0};
  int caculatedValue[2][2][2][2][2][2][2][2][2][2][2][2] = {0};
  int policyTable[2][2][2][2][2][2][2][2][2][2][2][2] = {0};

};

DotsAndBoxes::DotsAndBoxes(int mapsize) {
  mapSize_ = mapsize;
  mapTotalCount_ = mapSize_;

  map_.setZero(mapsize);
  initTerminateStates();
  initRewardStates();
}

void DotsAndBoxes::policyEvaluation(bool isFirst){
    Eigen::VectorXi map;
  map.setZero(mapSize_);
  for(int a = 0; a<2; a++){
  for(int b = 0; b<2; b++){
  for(int c = 0; c<2; c++){
  for(int d = 0; d<2; d++){
  for(int e = 0; e<2; e++){
  for(int f = 0; f<2; f++){
  for(int g = 0; g<2; g++){
  for(int h = 0; h<2; h++){
  for(int i = 0; i<2; i++){
  for(int j = 0; j<2; j++){
  for(int k = 0; k<2; k++){
  for(int l = 0; l<2; l++){
    map << a, b, c, d, e, f, g, h, i, j, k, l;
    takeActions(map, TakeActionType::EVALUATION, isFirst);
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
}

void DotsAndBoxes::policyImprovement(bool isFirst){
    Eigen::VectorXi map;
  map.setZero(mapSize_);
  for(int a = 0; a<2; a++){
  for(int b = 0; b<2; b++){
  for(int c = 0; c<2; c++){
  for(int d = 0; d<2; d++){
  for(int e = 0; e<2; e++){
  for(int f = 0; f<2; f++){
  for(int g = 0; g<2; g++){
  for(int h = 0; h<2; h++){
  for(int i = 0; i<2; i++){
  for(int j = 0; j<2; j++){
  for(int k = 0; k<2; k++){
  for(int l = 0; l<2; l++){
    map << a, b, c, d, e, f, g, h, i, j, k, l;
    takeActions(map, TakeActionType::IMPROVEMENT, isFirst);
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
}

double DotsAndBoxes::checkPolicyTerminateCondition(){
  double value=0.0;
  Eigen::VectorXi map;
  map.setZero(mapSize_);
  for(int a = 0; a<2; a++){
  for(int b = 0; b<2; b++){
  for(int c = 0; c<2; c++){
  for(int d = 0; d<2; d++){
  for(int e = 0; e<2; e++){
  for(int f = 0; f<2; f++){
  for(int g = 0; g<2; g++){
  for(int h = 0; h<2; h++){
  for(int i = 0; i<2; i++){
  for(int j = 0; j<2; j++){
  for(int k = 0; k<2; k++){
  for(int l = 0; l<2; l++){
    map << a, b, c, d, e, f, g, h, i, j, k, l;
    value+= getValue(map);
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }

  return value;
}

void DotsAndBoxes::train(){
  Eigen::VectorXi map, map1;
  double valueSum, preValueSum;
  valueSum=0.0; preValueSum=0.0;
  map.setZero(mapSize_);
  map1.setZero(mapSize_);
  policyEvaluation(true);
  map.setZero(mapSize_);
  policyImprovement(true);
  valueSum = checkPolicyTerminateCondition();
  if(abs(valueSum - preValueSum) < 1e-5)
  {
    return;
  }
  else{
    preValueSum = valueSum;
  }
  for(int i = 0 ; i<100; i++)
  {
    memset(caculatedValue, 0, sizeof(caculatedValue));
    map.setZero(mapSize_);
    policyEvaluation(false);
    map.setZero(mapSize_);
    policyImprovement(false);
    valueSum = checkPolicyTerminateCondition();
    if(abs(valueSum - preValueSum) < 1e-5)
    {
      return;
    }
    else{
      preValueSum = valueSum;
    }
  }
  return;

}

void DotsAndBoxes::printCurrentState(const Eigen::VectorXi &map) {
  std::cout << "\n==\t==\t==" << std::endl;
  std::cout << "o";
  if(map(0) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o";
  if(map(1) == 1) std::cout << " ㅡ " ;
  else std::cout << "   " ;
  std::cout << "o" << std::endl;
  if(map(6) == 1) std::cout << "|    " ;
  else std::cout << "    " ;
  if(map(8) == 1) std::cout << "|   " ;
  else std::cout << "    " ;
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
  else std::cout << "    " ;
  if(map(9) == 1) std::cout << "|    " ;
  else std::cout << "    " ;
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

void DotsAndBoxes::setState(const Eigen::VectorXi &map) {
  map_ = map;
}

void DotsAndBoxes::setValue(const Eigen::VectorXi &map, double value)
{
  valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = value;
}

void DotsAndBoxes::setPolicy(const Eigen::VectorXi &map, int policy)
{
  policyTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = policy;
}

void DotsAndBoxes::setCaculatedValue(const Eigen::VectorXi &map, int iscaculated)
{
  caculatedValue[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)] = iscaculated;
}

double DotsAndBoxes::getValue(const Eigen::VectorXi &map) {
  return valueTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
}

int DotsAndBoxes::getPolicy(const Eigen::VectorXi &map)
{
  return policyTable[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
}

int DotsAndBoxes::getCaculatedValue(const Eigen::VectorXi &map) {
  return caculatedValue[(int)map(0)][(int)map(1)][(int)map(2)][(int)map(3)][(int)map(4)][(int)map(5)][(int)map(6)][(int)map(7)][(int)map(8)][(int)map(9)][(int)map(10)][(int)map(11)];
}

//Should be changed when mapsize become different
void DotsAndBoxes::initTerminateStates() {
  Eigen::VectorXi terminateState;
  terminateState.setZero(mapTotalCount_);

  terminateState << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  terminateStates.push_back(terminateState);
}

void DotsAndBoxes::initRewardStates() {
  Eigen::VectorXi rewardState;
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

DotsAndBoxes::Type DotsAndBoxes::checkTermination(const Eigen::VectorXi map) {

  for(auto terminateState:terminateStates){
    int markSum = (map.array() * terminateState.array()).sum();
    if(markSum == mapSize_){
      return DotsAndBoxes::Type::TERMINATE;
    }
    else
      return DotsAndBoxes::Type::KEEP_GO;
    }
}


double DotsAndBoxes::giveReward(const Eigen::VectorXi &map, const Eigen::VectorXi &preMap) {

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

Eigen::VectorXi DotsAndBoxes::checkEmptyState(const Eigen::VectorXi &map) {
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

int DotsAndBoxes::getActionFromPolicy(const Eigen::VectorXi &map, bool isFirst){
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

double DotsAndBoxes::takeActions(const Eigen::VectorXi &map, TakeActionType takeActionType, bool isFirst){

  Eigen::VectorXi mapEmptyIndex;
  DotsAndBoxes::Type nextStateType;
  double nextStateValue = 0.0, reward = 0.0;
  Eigen::VectorXd tempValue;
  Eigen::VectorXi actionMap;
  mapEmptyIndex = checkEmptyState(map);
  tempValue.setZero(mapEmptyIndex(mapTotalCount_));
  actionMap.setZero(mapTotalCount_);
  nextStateType = checkTermination(map);

  if (nextStateType != DotsAndBoxes::Type::KEEP_GO) {  //Check Termination and give Value
    setCaculatedValue(map, 1);
    return 0.0;
  }
  else {
    if (takeActionType == TakeActionType::EVALUATION) { //Fixed policy, get Value function
      actionMap = map;
      actionMap(getActionFromPolicy(map, isFirst)) = 1.0;
      reward = giveReward(actionMap, map); // Caculate Reward
      if (reward > 1e-4) {
         if(getCaculatedValue(actionMap)) nextStateValue = getValue(actionMap);
         else nextStateValue = takeActions(actionMap, takeActionType, isFirst);
      }
      else nextStateValue = takeOpponentActionAndGetRewardValue(actionMap, takeActionType, isFirst); //Get v(s') by recursive function

      setValue(map, reward + gamma_ * nextStateValue);
      setCaculatedValue(map, 1);

//      std::cout << count_ << std::endl;
      return reward + gamma_ * nextStateValue;
    }
    else if (takeActionType == TakeActionType::IMPROVEMENT) { //Improve policy with q value
      for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) { //Iterate actions for all empty grid
        actionMap = map;
        actionMap(mapEmptyIndex(i)) = 1.0; //q function (s, a)
        reward = giveReward(actionMap, map); // Caculate Reward

        if (reward > 1e-4){
          tempValue(i) = reward + gamma_ * getValue(actionMap); }// Do Action Again
        else tempValue(i) = reward + gamma_ * takeOpponentActionAndGetRewardValue(actionMap, takeActionType, isFirst); //Get v(s') by recursive function
      }

      double maxValue, maxValueIndex;
      for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) {  //Improve policy with maximum state action value
        if (i == 0) {
          maxValue = tempValue(i);
          maxValueIndex = mapEmptyIndex(i);
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

double DotsAndBoxes::takeOpponentActionAndGetRewardValue(const Eigen::VectorXi &actionMap, TakeActionType takeActionType, bool isFirst) {
  DotsAndBoxes::Type nextStateType;
  Eigen::VectorXi mapEmptyIndex;
  Eigen::VectorXi nextMap;
  double value = 0.0, oppenentReward = 0.0;

  nextMap.setZero(mapTotalCount_);
  nextStateType = checkTermination(actionMap);

  if(nextStateType != DotsAndBoxes::Type::KEEP_GO){  //Terminate when user finished game. give value because we should give value for next state and multiply discount factor
    return 0.0;
  }
  else {
    mapEmptyIndex = checkEmptyState(actionMap);

    for (int i = 0; i < mapEmptyIndex(mapTotalCount_); i++) { //caculate value for all opponent action
      nextMap = actionMap;
      nextMap(mapEmptyIndex(i)) = 1.0;
      oppenentReward = giveReward(nextMap, actionMap);

      if (oppenentReward > 1e-4) value += takeOpponentActionAndGetRewardValue(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_);
      else{
          if(!getCaculatedValue(nextMap)) {
            value += takeActions(nextMap, takeActionType, isFirst) / mapEmptyIndex(mapTotalCount_); //getValue(nextMap) / mapEmptyIndex(mapTotalCount_);//
            setCaculatedValue(nextMap, 1);
          }
          else value += getValue(nextMap) / mapEmptyIndex(mapTotalCount_);//
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

  return dotsAndBoxes.getValue(state);  // return optimal value
}

/// DO NOT CHANGE THE NAME AND FORMAT OF THIS FUNCTION
int getOptimalAction(const Eigen::Vector<int, 12>& state){
  // return one of the optimal actions given the state.
  // the action should be represented as a state index, at which a line will be drawn.
  /// TODO
  DotsAndBoxes dotsAndBoxes(12);

  dotsAndBoxes.train();

  return dotsAndBoxes.getPolicy(state);  // return optimal action
}