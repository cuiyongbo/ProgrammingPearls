#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <atomic>

// https://github.com/sparsehash/sparsehash/blob/master/doc/performance.html

using namespace std;

#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1

#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"
#include "sparsehash/dense_hash_map"
#include "sparsehash/sparse_hash_map"
#include "timing_util.h"

using GOOGLE_NAMESPACE::dense_hash_map;
using GOOGLE_NAMESPACE::sparse_hash_map;


template<typename K, typename V>
class EasyUseDenseHashMap : public google::dense_hash_map<K,V> {
 public:
  EasyUseDenseHashMap() {
    this->set_empty_key(-1);
    this->set_deleted_key(-2);
  }
};

void shuffle(std::vector<int>& v) {
  srand(9);
  for (int n = v.size(); n >= 2; n--) {
    std::swap(v[n - 1], v[static_cast<unsigned>(rand()) % n]);
  }
}

template<typename MapType>
void profile_map(std::string label, int map_size, bool random_search) {
    MapType x;
    std::vector<int> indices;
    for (int i=0; i<map_size; ++i) {
        indices.push_back(i);
        x[i] = i;
    }

    if (random_search) {
        shuffle(indices);
    }

    TIMER_START(hash_map_test);
    std::atomic_int succCount(0);
    std::atomic_int failureCount(0);
    for (int i=0; i<map_size; ++i) {
        if (x.find(i) == x.end()) {
            ++failureCount;
        } else {
            ++succCount;
        }
    }
    //cout << label << " size: " << x.size() << ", success count: " << succCount << ", failure count: " << failureCount << endl;
    TIMER_STOP(hash_map_test);

    cout << label << ", random_search: " << random_search <<
        ", map_size: " << map_size <<
        ", using " << TIMER_MSEC(hash_map_test) << " ms, "
        << "avg latency: " << TIMER_NSEC(hash_map_test)/map_size << " ns"<< endl;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " map_size" << endl;
        return 1;
    }
    int map_size = std::stoi(argv[1]);
    profile_map<std::map<int, int>>("std::map", map_size, false);
    profile_map<std::unordered_map<int, int>>("std::unordered_map", map_size, false);
    profile_map<tbb::concurrent_unordered_map<int, int>>("tbb::concurrent_unordered_map", map_size, false);
    profile_map<EasyUseDenseHashMap<int, int>>("google::dense_hash_map", map_size, false);
    profile_map<google::sparse_hash_map<int, int>>("google::sparse_hash_map", map_size, false);
    profile_map<std::map<int, int>>("std::map", map_size, true);
    profile_map<std::unordered_map<int, int>>("std::unordered_map", map_size, true);
    profile_map<tbb::concurrent_unordered_map<int, int>>("tbb::concurrent_unordered_map", map_size, true);
    profile_map<EasyUseDenseHashMap<int, int>>("google::dense_hash_map", map_size, true);
    profile_map<google::sparse_hash_map<int, int>>("google::sparse_hash_map", map_size, true);

    return 0;
}

/*
$ ./profile_map_op  1000000
std::map, random_search: 0, map_size: 1000000, using 152.975 ms, avg latency: 152 ns
std::unordered_map, random_search: 0, map_size: 1000000, using 13.0178 ms, avg latency: 13 ns
tbb::concurrent_unordered_map, random_search: 0, map_size: 1000000, using 239.706 ms, avg latency: 239 ns
google::dense_hash_map, random_search: 0, map_size: 1000000, using 8.27348 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 1000000, using 18.341 ms, avg latency: 18 ns
std::map, random_search: 1, map_size: 1000000, using 136.004 ms, avg latency: 136 ns
std::unordered_map, random_search: 1, map_size: 1000000, using 38.8898 ms, avg latency: 38 ns
tbb::concurrent_unordered_map, random_search: 1, map_size: 1000000, using 249.092 ms, avg latency: 249 ns
google::dense_hash_map, random_search: 1, map_size: 1000000, using 8.24216 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 1000000, using 18.1757 ms, avg latency: 18 ns
*/