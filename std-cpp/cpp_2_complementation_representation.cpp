#include <iostream>
#include <bitset>

using namespace std;

using int32_bitset = std::bitset<32>;

int32_bitset twos_complement_representation(int num) {
  if (num >= 0) {
    return int32_bitset(num);
  } else {
    return int32_bitset(~(-num) + 1);
  }
}

int main() {
  int num;
  std::cout << "Enter an integer: ";
  std::cin >> num;

  int32_bitset result = twos_complement_representation(num);
  std::cout << "Two's complement representation of " << num << " is: " << result << std::endl;

  return 0;
}