#include <iostream>
#include <random>
#include <chrono>
#include <thread>

std::string generateRandomString(int len) {
  static std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(0, alphabet.size()-1);
  std::string result;
  for (int i=0; i<len; ++i) {
    result += alphabet[dist(rng)];
  }
  return result;
}

int main() {
  for (int i=1; i<100; i+=6) {
    std::cout << generateRandomString(i) << std::endl;
  }
}