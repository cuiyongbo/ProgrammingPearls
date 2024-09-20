#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <atomic>

// https://github.com/sparsehash/sparsehash/blob/master/doc/performance.html

/*
in case there is something wrong when building `sparsehash/sparsehash`, you may

1. change to environment with intel x86_64 cpu

2. try following commands

```bash
apt install autotools-dev automake make 
autoreconf -if
./configure
make # no more problem
make install
```
*/

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
    int loop_count = min(map_size, 10000);
    for (int i=0; i<loop_count; ++i) {
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
    srand(123456);
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
# lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
Address sizes:       40 bits physical, 48 bits virtual
CPU(s):              8
On-line CPU(s) list: 0-7
Thread(s) per core:  1
Core(s) per socket:  8
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz
Stepping:            7
CPU MHz:             2399.998
BogoMIPS:            4799.99
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            36608K
NUMA node0 CPU(s):   0-7
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat umip pku ospke avx512_vnni md_clear arch_capabilities

# ./profile_map_op 100000
std::map, random_search: 0, map_size: 100000, using 9.73565 ms, avg latency: 97 ns
std::unordered_map, random_search: 0, map_size: 100000, using 1.30846 ms, avg latency: 13 ns
google::dense_hash_map, random_search: 0, map_size: 100000, using 0.825718 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 100000, using 1.82348 ms, avg latency: 18 ns
std::map, random_search: 1, map_size: 100000, using 9.00196 ms, avg latency: 90 ns
std::unordered_map, random_search: 1, map_size: 100000, using 1.12313 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 1, map_size: 100000, using 0.813831 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 100000, using 1.83685 ms, avg latency: 18 ns

# ./profile_map_op 1000000
std::map, random_search: 0, map_size: 1000000, using 150.592 ms, avg latency: 150 ns
std::unordered_map, random_search: 0, map_size: 1000000, using 11.3772 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 0, map_size: 1000000, using 8.18561 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 1000000, using 17.9115 ms, avg latency: 17 ns
std::map, random_search: 1, map_size: 1000000, using 132.342 ms, avg latency: 132 ns
std::unordered_map, random_search: 1, map_size: 1000000, using 11.3916 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 1, map_size: 1000000, using 8.18782 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 1000000, using 17.9311 ms, avg latency: 17 ns

# ./profile_map_op 10000000
std::map, random_search: 0, map_size: 10000000, using 1813.11 ms, avg latency: 181 ns
std::unordered_map, random_search: 0, map_size: 10000000, using 115.527 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 0, map_size: 10000000, using 82.3605 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 10000000, using 183.535 ms, avg latency: 18 ns
std::map, random_search: 1, map_size: 10000000, using 1765.83 ms, avg latency: 176 ns
std::unordered_map, random_search: 1, map_size: 10000000, using 114.25 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 1, map_size: 10000000, using 82.1862 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 10000000, using 181.819 ms, avg latency: 18 ns

# ./profile_map_op 20000000
std::map, random_search: 0, map_size: 20000000, using 3805.64 ms, avg latency: 190 ns
std::unordered_map, random_search: 0, map_size: 20000000, using 228.882 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 0, map_size: 20000000, using 165.188 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 20000000, using 369.253 ms, avg latency: 18 ns
std::map, random_search: 1, map_size: 20000000, using 3693.9 ms, avg latency: 184 ns
std::unordered_map, random_search: 1, map_size: 20000000, using 229.938 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 1, map_size: 20000000, using 164.22 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 20000000, using 365.596 ms, avg latency: 18 ns

# ./profile_map_op 40000000
std::map, random_search: 0, map_size: 40000000, using 7954.21 ms, avg latency: 198 ns
std::unordered_map, random_search: 0, map_size: 40000000, using 462.392 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 0, map_size: 40000000, using 331.376 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 0, map_size: 40000000, using 734.66 ms, avg latency: 18 ns
std::map, random_search: 1, map_size: 40000000, using 7375.26 ms, avg latency: 184 ns
std::unordered_map, random_search: 1, map_size: 40000000, using 459.514 ms, avg latency: 11 ns
google::dense_hash_map, random_search: 1, map_size: 40000000, using 332.059 ms, avg latency: 8 ns
google::sparse_hash_map, random_search: 1, map_size: 40000000, using 737.781 ms, avg latency: 18 ns
*/