#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <atomic>

using namespace std;

#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1

//#include "tbb/concurrent_map.h"
//#include "tbb/concurrent_set.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "timing_util.h"
#include "util.h"

void usage() {
    cout << "Usage: prog mapSize\n";
}

void std_set_test(int mapSize);
void tbb_concurrent_set_test(int mapSize);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        usage();
        return EXIT_FAILURE;
    }

    int mapSize = std::atoi(argv[1]);
    std_set_test(mapSize);
    tbb_concurrent_set_test(mapSize);
}

void std_set_test(int mapSize) {
    TIMER_START(std_set_test);
    std::set<int> x;
    for (int i=0; i<mapSize; ++i) {
        x.insert(i);
    }
    cout << "std::set size: " << x.size() << endl;
    int succCount = 0;
    int failureCount = 0;
    for (int i=0; i<mapSize; ++i) {
        if (x.find(i) == x.end()) {
            ++failureCount;
        } else {
            ++succCount;
        }
    }
    cout << "success count: " << succCount << ", failure count: " << failureCount << endl;
    TIMER_STOP(std_set_test);
    cout << "std::set test using " << TIMER_MSEC(std_set_test) << " milliseconds, "
         << "avg latency: " << TIMER_NSEC(std_set_test)/mapSize << " ns"<< endl;
}

void tbb_concurrent_set_test(int mapSize) {
    TIMER_START(tbb_concurrent_set_test);
    tbb::concurrent_set<int> x;
    tbb::parallel_for (tbb::blocked_range<int>(0, mapSize), 
        [&](const tbb::blocked_range<int>& range) {
            for (int i=range.begin(); i != range.end(); ++i) {
                x.insert(i);
            }
        }
    );
    cout << "tbb::concurrent_set size: " << x.size() << endl;
    std::atomic_int succCount(0);
    std::atomic_int failureCount(0);
    tbb::parallel_for (tbb::blocked_range<int>(0, mapSize), 
        [&](const tbb::blocked_range<int>& range) {
            for (int i=range.begin(); i != range.end(); ++i) {
                if (x.find(i) == x.end()) {
                    ++failureCount;
                } else {
                    ++succCount;
                }
            }
        }
    );
    cout << "success count: " << succCount << ", failure count: " << failureCount << endl;
    TIMER_STOP(tbb_concurrent_set_test);

    cout << "tbb::concurrent_set test using " << TIMER_MSEC(tbb_concurrent_set_test) << " milliseconds, "
         << "avg latency: " << TIMER_NSEC(tbb_concurrent_set_test)/mapSize << " ns"<< endl;
}