#include <iostream>
#include <atomic>
#include <thread>

std::atomic<bool> flag(true);

void toggleFlag() {
    // Toggle the value of the atomic flag
    flag.store(!flag.load());
}

int main() {
    // Create two threads that toggle the flag
    std::thread thread1(toggleFlag);
    std::thread thread2(toggleFlag);

    // Wait for the threads to finish
    thread1.join();
    thread2.join();

    // Read the final value of the flag
    bool finalValue = flag.load();

    // Print the final value
    std::cout << "Final value of the flag: " << std::boolalpha << finalValue << std::endl;

    return 0;
}
