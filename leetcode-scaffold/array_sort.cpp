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

/*
// O(nlogn) on average case
1. pick and element, called a pivot, from the array
2. patitioning: reorder the array so that the elements with values less than the pivot come before the pivot, and the elements with values greater than the pivot come after it (equal values can go either way). after the partition, the pivot is in its final position
3. recursively apply the above steps to all sub-arrays
*/
void Solution::quickSort(vector<int>& nums) {
    // l, r are inclusive
    auto naive_partitioner = [&] (int l, int r) {
        int i = l-1;
        int pivot = nums[r];
        for (int k=l; k<r; ++k) {
            if (nums[k] < pivot) { // num[k]<pivot, l <= k <= i
                std::swap(nums[++i], nums[k]);
            }
        }
        // move pivot to its final sorted position
        std::swap(nums[i+1], nums[r]);
        return i+1;
    };
    // l, r are inclusive
    std::function<void(int, int)> dac = [&] (int l, int r) {
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

/*
// \Theta(log_b(k)*(n+k)), k is the maximum element value in the element

Radix-Sort(A, d)
    for i=1 to d
        use a stable sort to sort array A on i_th digit
*/
void Solution::radixSort(vector<int>& nums) {
    auto p = minmax_element(nums.begin(), nums.end());
    int l = *(p.first);
    int r = *(p.second);

    int digit_num = 0;
    int max_ele = r-l;
    do {
        max_ele/=10;
        digit_num++;
    } while (max_ele!=0);

    // map array elements to [0, r-l]
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](int n){return n-l;});

    auto get_digit = [](int n, int i) {
        for (; i!=0 && n!=0; i--) {
            n/=10;
        }
        return n%10;
    };

    for (int i=0; i<=digit_num; i++) {
        std::vector<vector<int>> buff(10);
        // sort by nth digit
        for (auto n: nums) {
            auto d = get_digit(n, i);
            buff[d].push_back(n);
        }
        nums.clear();
        for (const auto& t:  buff) {
            nums.insert(nums.end(), t.begin(), t.end());
        }
    }

    // remap array elements to original value
    std::transform(nums.begin(), nums.end(), nums.begin(), [&](int n){return n+l;});
}

// O(nlogn) for worst-case running time
void Solution::heapSort(vector<int>& nums) {

{ // std solution
    // in `std::priority_queue` template, we perform `compare(child, root)` test to see whether root, left-child, right-child are in heap-order or not
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq(nums.begin(), nums.end()); // minHeap
    nums.clear();
    while (!pq.empty()) {
        nums.push_back(pq.top());
        pq.pop();
    }
    return;
}

{ // naive solution
    // 1. build heap with sift-up
    // 2. pop heap head
    // 3. restore heap with sift-down
    // 4. repeat step 2 and 3 until the heap is empty
    std::function<void(int, int)> sift_down = [&] (int i, int sz) {
        // i; // root
        int largest = i;
        while (i < sz) {
            largest = i;
            int li = 2*i+1; // left child
            if (li<sz && nums[li]>nums[largest]) {
                largest = li;
            }
            int ri = 2*i+2; // right child
            if (ri<sz && nums[ri]>nums[largest]) {
                largest = ri;
            }
            if (largest == i) {
                break;
            }
            swap(nums[largest], nums[i]);
            i = largest;
        }
    };
    int sz = nums.size();
    // 1. build heap with sift-up
    for (int i=(sz-1)/2; i>=0; i--) {
        sift_down(i, sz);
    }
    for (int i=sz-1; i>0; i--) {
        // 2. pop heap head
        swap(nums[0], nums[i]);
        // 3. restore heap with sift-down
        sift_down(0, i);
    }
    return;
}

}

/*
// O(nlogn) for worst-case running time
1. divide the unsorted list into n sublists, each containing 1 element
2. repeatedly merge sublists to produce new sorted sublists untill there is only 1 sublist remaining. this will be the sorted list
*/
void Solution::mergeSort(std::vector<int>& nums) {
    std::vector<int> twins = nums;
    std::function<void(int, int)> dac = [&] (int l, int r) {
        if (l >= r) { // trivial case
            return;
        }
        int m = (l+r)/2;
        // divide
        dac(l, m); dac(m+1, r);
        // conquer
        int i=l;
        int j=m+1;
        for (int k=l; k<=r; k++) {
            if ((i<=m && nums[i]<nums[j]) || (j>r)) {
                twins[k] = nums[i++];
            } else {
                twins[k] = nums[j++];
            }
        }
        // swap elements in [l, r]
        std::copy(twins.begin()+l, twins.begin()+r+1, nums.begin()+l);
    };
    dac(0, nums.size()-1);
}

// \Theta(nlogn)
void Solution::bstSort(vector<int>& nums) {
    // std::multiset maybe implemented with red-black tree
    std::multiset<int> s(nums.begin(), nums.end());
    std::copy(s.begin(), s.end(), nums.begin());
}

// \Theta(n^2)
void Solution::insertionSort(vector<int>& nums) {
    for (int i=1; i<nums.size(); ++i) {
        // premise: array [0, i-1] is sorted
        // 1. find temporary sorted position for nums[i]
        for (int j=0; j<i; ++j) {
            if (nums[j] > nums[i]) { // make sure it is a stable sort algorithm
                // 2. then shift array[j, i-1] towards the right direction by one, and move nums[i] to nums[j]
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
    std::random_device rd;
    std::mt19937 g(rd());
    Solution ss;
    vector<int> vi; vi.reserve(array_size);
    for (int i=0; i<array_size; i++) {
        vi.push_back(rand());
    }
    for (int i=0; i<100; ++i) {
        int n = rand() % array_size;
        std::shuffle(vi.begin(), vi.begin()+n, g);
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
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_radixSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_mergeSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_quickSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_heapSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_insertionSort);
    //batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_countingSort);
    TIMER_STOP(sortArray_batch_test);
    util::Log(logESSENTIAL) << "batch tests using " << TIMER_MSEC(sortArray_batch_test) << " milliseconds";
}
