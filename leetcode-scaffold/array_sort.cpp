#include "leetcode.h"

using namespace std;

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

private:
    void quick_sort_worker(vector<int>& nums, int l, int r);
    int quick_sort_partitioner(vector<int>& nums, int l, int r);
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
        // randomly choose pivots in case that the algorithm degrades when the array is already sorted or nearly sorted
        int rr = rand()%(r-l+1) + l;
        swap(nums[rr], nums[r]);
        int i = l-1;
        int pivot = nums[r];
        for (int k=l; k<r; ++k) {
            if (nums[k] < pivot) { // num[k]<pivot, l <= k <= i
                ++i;
                std::swap(nums[i], nums[k]);
            }
        }
        // move pivot to its final sorted position
        ++i;
        std::swap(nums[i], nums[r]);
        return i;
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
    long range = r-l+1;
    SPDLOG_INFO("CountingSort(min={}, max={}, range={})", l, r, range);
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

/*
two loops:

outer loop: extract the largest from heap top at each step
inner loop: reoragnize the remaining array, so it doesn't invalidate the heap property


*/

{

    auto sift_down = [&] (int root, int size) {
        while (root<size) {
            int p_i = root;
            int left=2*p_i+1;
            int right=2*p_i+2;
            if (left<size && nums[left]>nums[p_i]) {
                p_i = left;
            }
            if (right<size && nums[right]>nums[p_i]) {
                p_i = right;
            }
            if (p_i == root) {
                break;
            }
            swap(nums[p_i], nums[root]);
            root = p_i;
        }
    };
    // 1. build heap with sift-up
    int size = nums.size();
    for (int i=size/2; i>=0; i--) {
        sift_down(i, size);
    }

    // 2. extract the largest element from the remaining array one by one
    for (int i=size-1; i>0; i--) {
        std::swap(nums[0], nums[i]);
        // sift-down
        sift_down(0, i);
    }
    return;
}



{ // std solution
    // in `std::priority_queue` template, we perform `compare(child, root)` test to see whether root, left-child, right-child are in heap-order or not
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap(nums.begin(), nums.end()); // minHeap
    nums.clear();
    while (!min_heap.empty()) {
        nums.push_back(min_heap.top());
        min_heap.pop();
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
    /*
        [l, r), i
        [l, i] is sorted in ascending order
        [i+1, r) is not sorted
        the algorithm consists of two loops:
            loop one: expand the sorted array to right by one at each iteration
            loop two: make sure the left part is sorted
    */
    int l = 0;
    int r = nums.size();
    for (int i=l+1; i<r; i++) {
        int p = nums[i];
        for (int j=l; j<i; j++) {
            if (p < nums[j]) {
                // p should be placed at nums[j]
                // but before that we need shift nums[j, i-1] to right by one
                for (int k=i; k>j; k--) {
                    nums[k] = nums[k-1];
                }
                nums[j] = p;
                break;
            }
        }
    }
    return;
}

void sortArray_scaffold(string input, AlgorithmType type) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    ss.sortArray(vi, type);
    if(std::is_sorted(vi.begin(), vi.end())) {
        SPDLOG_INFO("case({}, {}) passed", input, AlgorithmType_toString(type));
    } else {
        SPDLOG_INFO("case({}, {}) failed", input, AlgorithmType_toString(type));
    }
}

void batch_test_scaffold(int array_scale, AlgorithmType type) {
    SPDLOG_INFO("Running {} tests", AlgorithmType_toString(type));
    std::random_device rd;
    std::mt19937 g(rd());
    Solution ss;
    vector<int> vi; vi.reserve(array_scale);
    for (int i=0; i<array_scale; i++) {
        vi.push_back(rand());
    }
    for (int i=0; i<100; ++i) {
        SPDLOG_INFO("Running {} tests at {}", AlgorithmType_toString(type), i+1);
        int n = rand() % array_scale;
        std::shuffle(vi.begin(), vi.begin()+n, g);
        ss.sortArray(vi, type);
        if(!std::is_sorted(vi.begin(), vi.end())) {
            SPDLOG_ERROR("Case(array_scale={}, array_size={}, algorithm={}) failed", array_scale, n, AlgorithmType_toString(type));
        }
    }
}

void basic_test() {
    SPDLOG_INFO("Running sortArray tests:");
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
    SPDLOG_INFO("sortArray using {:.2f} milliseconds", TIMER_MSEC(sortArray));
}

int main(int argc, char* argv[]) {
    //basic_test();

    int array_size = 100;
    if (argc > 1) {
        array_size = std::atoi(argv[1]);
        if (array_size <= 0) {
            SPDLOG_WARN("Usage: {} [arrary_size]", argv[0]);
            SPDLOG_WARN("\tarrary_size must be positive, default to 100 if unspecified");
            return -1;
        }
    }

    SPDLOG_INFO("Running batch tests(array_size={})", array_size);
    TIMER_START(sortArray_batch_test);
    //batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_countingSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_radixSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_mergeSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_quickSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_heapSort);
    batch_test_scaffold(array_size, AlgorithmType::AlgorithmType_insertionSort);
    TIMER_STOP(sortArray_batch_test);
    SPDLOG_INFO("batch tests using {:.2f} milliseconds", TIMER_MSEC(sortArray_batch_test));
}
