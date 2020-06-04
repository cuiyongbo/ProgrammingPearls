#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercise: 912 */

enum class AlgorithmType
{
    AlgorithmType_none,
    AlgorithmType_quickSort,
    AlgorithmType_countingSort,
    AlgorithmType_heapSort,
    AlgorithmType_mergeSort,
    AlgorithmType_binarySearchTree,
};

const char* AlgorithmType_toString(AlgorithmType type)
{
    const char* str = "invalid algorithm type";
    switch (type)
    {
    case AlgorithmType::AlgorithmType_quickSort:
        str = "quickSort";
        break;
    case AlgorithmType::AlgorithmType_countingSort:
        str = "coutingSort";
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
    default:
        break;
    }
    return str;
}

class Solution 
{
public:
    void sortArray(vector<int>& nums); 
    void quickSort(vector<int>& nums);
    void countingSort(vector<int>& nums);
    void heapSort(vector<int>& nums);
    void mergeSort(vector<int>& nums);
    void bstSort(vector<int>& nums);
};

void Solution::sortArray(vector<int>& nums)
{
    quickSort(nums);
}

// O(nlogn) on average case
void Solution::quickSort(vector<int>& nums)
{
    function<void(int, int)> dac = [&] (int l, int r)
    {
        // trivial case
        if(l+1 >= r) return;

        int i = l-1;
        int k = nums[r-1];
        for(int j=l; j<r; ++j)
        {
            if(nums[j] < k)
            {
                swap(nums[j], nums[++i]);
            }
        }

        // put k in its final position when the array is sorted
        swap(nums[i+1], nums[r-1]);

        int m = i+1;
        dac(l, m);
        dac(m+1, r);
    };

    dac(0, nums.size());
}

// \Theta(k) [k = max(array) - min(array)]
void Solution::countingSort(vector<int>& nums)
{
    /* 
        NOT suitable for sparse arrays splitting in a large array, 
        such as [INT_MIN, INT_MAX], suffering to overflow problem
    */

    if(nums.size() < 2) return;

    vector<int>::iterator minIt, maxIt;
    std::tie(minIt, maxIt) = minmax_element(nums.begin(), nums.end());
    int l = *minIt, r = *maxIt;
    vector<int> count(r-l+1, 0);
    for(auto n: nums)  ++count[n-l];

    int curCount = 0;
    for(int i=0; i<count.size(); ++i)
    {
        for(int j=0; j<count[i]; ++j)
            nums[curCount++] = l + i;  
    }
}

// O(nlogn) for worst-case running time
void Solution::heapSort(vector<int>& nums)
{
    // minHeap
    priority_queue<int, vector<int>, std::greater<int>> pq(nums.begin(), nums.end());
    int curIndex = 0;
    while(!pq.empty())
    {
        nums[curIndex++] = pq.top();
        pq.pop();
    }
}

// O(nlogn) for worst-case running time
void Solution::mergeSort(vector<int>& nums)
{
    vector<int> twin = nums;

    auto merger = [&](int l, int m, int r)
    {
        int k = l;
        int i=l, j=m+1;
        while(i<=m || j<=r)
        {
            if( (j > r) || 
                ((i<=m) && nums[i] < nums[j]))
            {
                twin[k++] = nums[i++];
            }
            else
            {
                // j <= r
                twin[k++] = nums[j++];
            }
        }

        std::copy(begin(twin)+l, begin(twin)+r+1, begin(nums)+l);
    };

    function<void(int, int)> dac = [&] (int l, int r)
    {
        // trivial case
        if(l >= r) return;

        int m = l + (r-l)/2;
        dac(l, m); 
        dac(m+1, r);

        merger(l, m ,r); // nums[l, r] is sorted when return
    };

    dac(0, nums.size()-1);
}

// \Theta(nlogn)
void Solution::bstSort(vector<int>& nums)
{
    multiset<int> s(nums.begin(), nums.end());
    std::copy(s.begin(), s.end(), nums.begin());
}

void sortArray_scaffold(string input, AlgorithmType type)
{
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    switch(type)
    {
        case AlgorithmType::AlgorithmType_quickSort:
            ss.quickSort(vi);
            break;
        case AlgorithmType::AlgorithmType_countingSort:
            ss.countingSort(vi);
            break;
        case AlgorithmType::AlgorithmType_heapSort:
            ss.heapSort(vi);
            break;
        case AlgorithmType::AlgorithmType_mergeSort:
            ss.mergeSort(vi);
            break;
        case AlgorithmType::AlgorithmType_binarySearchTree:
            ss.bstSort(vi);
            break;
        default:
            util::Log(logWARNING) << "invalid AlgorithmType";
            return;
    }

    if(std::is_sorted(vi.begin(), vi.end()))
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << AlgorithmType_toString(type) << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << AlgorithmType_toString(type) << ") passed";
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

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

    sortArray_scaffold("[1,3,2,4,6,5]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[6,5,4,3,2,1]", AlgorithmType::AlgorithmType_binarySearchTree);
    sortArray_scaffold("[1,1,1,1,1,1]", AlgorithmType::AlgorithmType_binarySearchTree);

    TIMER_STOP(sortArray);
    util::Log(logESSENTIAL) << "sortArray using " << TIMER_MSEC(sortArray) << " milliseconds\n";
}