#include <chrono>
#include <iostream>

constexpr int kSize = 100000;
int data[kSize][kSize];

int test_sum_v0() {
  int sum = 0;
  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      sum += data[i][j];
    }
  }
  return sum;
}

int test_sum_v1() {
  int sum = 0;
  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      sum += data[j][i];
    }
  }
  return sum;
}

int main() {
  std::srand(std::time(nullptr));
  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      data[i][j] = std::rand();
    }
  }

  // 奇怪, 这两种方式耗时差别不大, 甚至第二种耗时会更小
  auto start = std::chrono::steady_clock::now();
  int sum0 = test_sum_v0();
  auto cost = std::chrono::steady_clock::now() - start;
  std::cout << "sum0=" << sum0 << ", cost=" << cost.count() << std::endl;

  start = std::chrono::steady_clock::now();
  int sum1 = test_sum_v1();
  cost = std::chrono::steady_clock::now() - start;
  std::cout << "sum1=" << sum1 << ", cost=" << cost.count() << std::endl;

  return 0;
}


