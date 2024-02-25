#include <iostream>
#include <random>
#include <chrono>
#include <thread>
// https://en.cppreference.com/w/cpp/numeric/random
int main() {
  std::random_device dev;
  std::mt19937 rng(dev()); // Random Number Generator
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  std::uniform_int_distribution<> dist(1,6); // distribution in range [1, 6]
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  //std::uniform_real_distribution<> dist(0, 1.0); // distribution in range [0, 1.0)
  while(true) {
    std::cout << dist(rng) << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}