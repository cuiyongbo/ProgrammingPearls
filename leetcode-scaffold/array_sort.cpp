#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercise: 912 */

enum AlgorithmType {
    AlgorithmType_none,
    AlgorithmType_quickSort,
    AlgorithmType_countingSort,
    AlgorithmType_radixSort,
    AlgorithmType_heapSort,
    AlgorithmType_mergeSort,
    AlgorithmType_binarySearchTree,
    AlgorithmType_insertionSort,
};

const char* AlgorithmType_toString(AlgorithmType type) {
    const char* str = nullptr;
    switch (type) {
    case AlgorithmType::AlgorithmType_quickSort:
        str = "quickSort";
        break;
    case AlgorithmType::AlgorithmType_countingSort:
        str = "countingSort";
        break;
    case AlgorithmType::AlgorithmType_radixSort:
        str = "radixSort";
        break;
    case AlgorithmType::AlgorithmType_heapSort:
        str = "heapSort";
        break;
    case AlgorithmType::AlgorithmType_mergeSort:
        str = "mergeSort";
        break;
    case AlgorithmType::AlgorithmType_binarySearchTree:
        str = "binarySearchTree";
        break;
    case AlgorithmType::AlgorithmType_insertionSort:
        str = "insertionSort";
        break;
    default:
        str = "unknown";
        break;
    }
    return str;
}

class Solution {
public:
    void sortArray(vector<int>& nums, AlgorithmType type=AlgorithmType_none); 
    void quickSort(vector<int>& nums);
    void countingSort(vector<int>& nums);
    void heapSort(vector<int>& nums);
    void mergeSort(vector<int>& nums);
    void bstSort(vector<int>& nums);
    void insertionSort(vector<int>& nums);
    void radixSort(vector<int>& nums);
};

void Solution::sortArray(vector<int>& nums, AlgorithmType type) {
    if (type == AlgorithmType_quickSort) {
        quickSort(nums);
    } else if (type == AlgorithmType_countingSort) {
        countingSort(nums);
    } else if (type == AlgorithmType_heapSort) {
        heapSort(nums);
    } else if (type == AlgorithmType_mergeSort) {
        mergeSort(nums);
    } else if (type == AlgorithmType_binarySearchTree) {
        bstSort(nums);
    } else if (type == AlgorithmType_insertionSort) {
        insertionSort(nums);
    } else if (type == AlgorithmType_radixSort) {
        radixSort(nums);
    } else {
        cout << "invalid algorithm type" << endl;
    }
}

// O(nlogn) on average case
void Solution::quickSort(vector<int>& nums) {
    auto naive_partitioner = [&] (int l, int r) {
        int i = l-1;
        int pivot = nums[r];
        for (int k=l; k<r; ++k) {
            if (nums[k] < pivot) { // num[k]<pivot, l <= k <= i
                std::swap(nums[++i], nums[k]);
            }
        }
        // put pivot in its final position when array is sorted
        std::swap(nums[i+1], nums[r]);
        return i+1;
    };
    function<void(int, int)> dac = [&] (int l, int r) {
        if (l >= r) { // trivial case
            return;
        }
        int m = naive_partitioner(l, r);
        dac(l, m-1);
        dac(m+1, r);
    };
    dac(0, nums.size()-1);
}

// \Theta(k) [k = max(array) - min(array)]
void Solution::countingSort(vector<int>& nums) {
/* 
    NOT suitable for sparse arrays splitting in a large array, 
    such as [INT_MIN, INT_MAX], suffering to range overflow problem
*/
    auto p = minmax_element(nums.begin(), nums.end());
    int l = *(p.first);
    int r = *(p.second);
    util::Log(logDEBUG) << "CountingSort(min=" << l << ", max=" << r << ")";
    long range = r-l+1;
    vector<int> count(range, 0);
    for (auto n: nums) {
        count[n-l]++;
    }
    nums.clear();
    for (long i=0; i<range; ++i) {
        if (count[i] != 0) {
            nums.insert(nums.end(), count[i], long(i+l));
        }
    }
    return;
}

// \Theta(log_b(k)*(n+k)), k is the maximum element value in the element
void Solution::radixSort(vector<int>& nums) {

    auto p = minmax_element(nums.begin(), nums.end());
    int l = *(p.first);
    int r = *(p.second);

    // map the value range [l, r] of elements in nums to [1, r-l+1]
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](const int& n) {return n-l+1;});

    int max = r-l+1;
    int digit_count = 0;
    do {
        max /= 10;
        digit_count++;
    } while (max!=0);

    auto get_digit = [] (int n, int i) {
        while (i!=0) {
            n/=10;
        }
        return n%10;
    };

    int radix = 10;
    std::vector<std::vector<int>> count(radix);
    for (int i=0; i<digit_count; ++i) {
        count.assign(radix, vector<int>());
        for (auto n: nums) {
            int digit = get_digit(n, i);
            count[digit].push_back(n);
        }
        nums.clear();
        for (auto& t: count) {
            nums.insert(nums.end(), t.begin(), t.end());
        }
    }

    // restore original value
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](const int& n) {return n+l-1;});
}

// O(nlogn) for worst-case running time
void Solution::heapSort(vector<int>& nums) {

{ // naive solution
    int sz = nums.size();
    // make max-heap: sift up
    for (int i=1; i<sz; ++i) {
        int l = i;
        while (true) {
            int p = (l-1)/2;
            if (nums[p]>=nums[l]) {
                break;
            }
            std::swap(nums[l], nums[p]);
            l = p;
        }
    }
    for (int i=0; i<sz-1; ++i) {
        // pop heap
        int r=sz-1-i;
        std::swap(nums[0], nums[r]);
        // restore heap: sift down
        int l=0;
        while (true) {
            int largest = l;
            int left = 2*l+1;
            if (left<r && nums[largest]<nums[left]) {
                largest = left;
            }
            int right = 2*l+2;
            if (right<r && nums[largest]<nums[right]) {
                largest = right;
            }
            if (largest == l) {
                break;
            }
            std::swap(nums[largest], nums[l]);
            l = largest;
        }
    }
    return;
}

{ // std solution
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq(nums.begin(), nums.end()); // minHeap
    nums.clear();
    while (!pq.empty()) {
        nums.push_back(pq.top());
        pq.pop();
    }
}

}

// O(nlogn) for worst-case running time
void Solution::mergeSort(std::vector<int>& nums) {
    std::vector<int> twins = nums;
    std::function<void(int, int)> dac = [&] (int l, int r) {
        if (l>=r) { // trivial case
            return;
        }
        int m = (l+r)/2;
        dac(l, m); dac(m+1, r);
        int k=l;
        int i=l;
        int j=m+1;
        while (i<=m || j<=r) {
            if (j>r || (i<=m&&nums[i]<nums[j])) {
                twins[k++] = nums[i++];
            } else {
                twins[k++] = nums[j++];
            }
        }
        // swap elements in [l, r]
        std::copy(twins.begin()+l, twins.begin()+r+1, nums.begin()+l);
    };
    dac(0, nums.size()-1);
    return;
}

// \Theta(nlogn)
void Solution::bstSort(vector<int>& nums) {
    multiset<int> s(nums.begin(), nums.end());
    std::copy(s.begin(), s.end(), nums.begin());
}

// \Theta(n^2)
void Solution::insertionSort(vector<int>& nums) {
    int size = nums.size();
    for (int i=1; i<size; ++i) {
        // premise: array [0, i-1] is sorted
        // 1. find temporary sorted position for nums[i]
        for (int j=0; j<i; ++j) {
            if (nums[j] > nums[i]) { // make sure it is a stable sort algorithm
                // 2. put nums[i] to the position
                int tmp = nums[i];
                for (int k=i; k!=j; --k) {
                    nums[k] = nums[k-1];
                }
                nums[j] = tmp;
                break;
            }
        }     
        // post-condition: array[0, i] is sorted
    }
}

void sortArray_scaffold(string input, AlgorithmType type) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    ss.sortArray(vi, type);
    if(std::is_sorted(vi.begin(), vi.end())) {
        util::Log(logINFO) << "Case(" << input << ", " << AlgorithmType_toString(type) << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << AlgorithmType_toString(type) << ") failed";
    }
}

void batch_test_scaffold(int array_size, AlgorithmType type) {
    util::Log(logINFO) << "Running " << AlgorithmType_toString(type) << " tests";
    Solution ss;
    vector<int> vi; vi.reserve(array_size);
    for (int i=0; i<100; ++i) {
        vi.clear();
        int n = rand() % array_size;
        for (int j=0; j<n; ++j) {
            vi.push_back(rand());
        }
        ss.sortArray(vi, type);
        if(!std::is_sorted(vi.begin(), vi.end())) {
            util::Log(logERROR) << "Case(array_size<" << array_size 
                << ">, array_size<" << n << ">, "  << AlgorithmType_toString(type) << ") failed";
        }
    }
}

void basic_test() {
    util::Log(logESSENTIAL) << "Running sortArray tests:";
    TIMER_START(sortArray);
    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_quickSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_quickSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_quickSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_countingSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_countingSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_countingSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_heapSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_heapSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_heapSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_mergeSort);
    sortArray_scaffold("[3,1,2]", AlgorithmType::AlgorithmType_mergeSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_binarySearchTree);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_insertionSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_insertionSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_insertionSort);

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_radixSort);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_radixSort);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_radixSort);

    TIMER_STOP(sortArray);
    util::Log(logESSENTIAL) << "sortArray using " << TIMER_MSEC(sortArray) << " milliseconds";
}

int main(int argc, char* argv[]) {
    util::LogPolicy::GetInstance().Unmute();

    //basic_test();

    int array_size = 100;
    if (argc > 1) {
        array_size = std::atoi(argv[1]);
        if (array_size <= 0) {
            printf("Usage: %s [arrary_size]\n", argv[0]);
            printf("\tarrary_size must be positive, default to 100 if unspecified\n");
            return -1;
        }
    }

    util::Log(logESSENTIAL) << "Running batch tests(array_size=" << array_size << "):";
    TIMER_START(sortArray_batch_test);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_mergeSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_quickSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_heapSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_insertionSort);
    //batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_countingSort);
    TIMER_STOP(sortArray_batch_test);
    util::Log(logESSENTIAL) << "batch tests using " << TIMER_MSEC(sortArray_batch_test) << " milliseconds";
}
