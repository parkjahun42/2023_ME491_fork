#include <iostream>
#include "student_id.hpp"

int main() {
  Eigen::Vector<int, 12> state;
  state << 0,0,0,0,0,0,0,0,0,0,0,1;
  std::cout << "optimal value for the state: " << getOptimalValue(state) << std::endl;
  std::cout << "optimal action for the state: " << getOptimalAction(state) << std::endl;

  return 0;
}
