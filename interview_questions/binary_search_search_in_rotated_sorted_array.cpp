#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 33, 81, 153, 154, 162, 852 */
class Solution 
{
public:
    int search_33(vector<int>& nums, int target);
    bool search_81(vector<int>& nums, int target);
    int findMin(vector<int> &num);
    int findPeakElement(vector<int>& nums);
    int peakIndexInMountainArray(vector<int>& A);
};

int Solution::search_33(vector<int>& nums, int target)
{
    /*
        Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
        (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).
        You are given a target value to search. If found in the array return its index, otherwise return -1.
        You may assume no duplicate exists in the array.
        Your algorithm's runtime complexity must be in the order of O(log n).
    */

    int l = 0;
    int r = (int)nums.size() - 1;
    while(l <= r)
    {
        int m = l + (r-l)/2;
        if(nums[m] == target)
        {
            return m;
        }
        else if(nums[l] < nums[r])
        {
            // [l, r] is sorted
            if(nums[m] < target)
            {
                l = m+1;
            }
            else
            {
                r = m-1;
            }
        }
        else
        {
            // [l, r] is rotated

            if(nums[l] < nums[m])
            {
                // [l, m] is sorted
                if(nums[l] <= target && target <= nums[m])
                {
                    r = m-1;
                }
                else
                {
                    l = m+1;
                }
            }
            else
            {
                // [m, r] is sorted
                if(nums[m] <= target && target <= nums[r])
                {
                    l = m+1;
                }
                else
                {
                    r = m-1;
                }
            }
        }
    }
    return -1;
}

bool Solution::search_81(vector<int>& nums, int target)
{
    /*
        Same with search_33 except that nums may contain duplicate
    */

    function<bool(int, int)> dac = [&](int l, int r)
    {
        if(l > r) return false;
        int m = l + (r-l)/2;
        if(nums[m] == target)
        {
            return true;
        }

        if(nums[l] < nums[r])
        {
            // [l, r] is sorted
            if(nums[m] < target)
                l = m+1;
            else
                r = m-1;

            return dac(l, r);
        }
        else
        {
            return dac(l, m-1) || dac(m+1, r);
        }
    };

    return dac(0, nums.size() - 1);
}

int Solution::findMin(vector<int>& nums)
{
    /*
        Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
        (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
        Find the minimum element.
        You may assume no duplicate exists in the array.
    */

    assert(!nums.empty() && "input array must NOT be empty");

    function<int(int, int)> dac = [&](int l, int r)
    {
        // trivial case
        if(l == r) return nums[l];
        
        // [l, r] is sorted
        if(nums[l] < nums[r])
            return nums[l];

        int m = l + (r-l)/2;
        int ml = dac(l, m);
        int mr = dac(m+1, r);
        return std::min(ml, mr);
    };

    return dac(0, (int)nums.size()-1);
}

int Solution::findPeakElement(vector<int>& nums)
{
    /*
        A peak element is an element that is greater than its neighbors.
        Given an input array nums, where nums[i] != nums[i+1], find a peak element and return its index.
        The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
        You may imagine that nums[-1] = nums[n] = -inf.
    */

    assert(!nums.empty() && "input array must NOT be empty");

    vector<int> cp(nums.size() + 2, INT32_MIN);
    std::copy(nums.begin(), nums.end(), cp.begin() + 1);

    function<int(int, int)> dac = [&](int l, int r)
    {
        if(l > r) return INT32_MAX;

        int m = l + (r-l)/2;
        if(cp[m-1] < cp[m] && cp[m] > cp[m+1])
            return m;

        int ml = dac(l, m-1);
        int mr = dac(m+1, r);
        return std::min(ml, mr);
    };

    return dac(1, nums.size()) - 1;
}

int Solution::peakIndexInMountainArray(vector<int>& nums)
{
    /*
        Let’s call an array A a mountain if the following properties hold:

        A.length >= 3
        There exists some 0 < i < A.length - 1 such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1]
        Given an array that is definitely a mountain, return such index.
    */

    assert(nums.size() >= 3 && "input array size must NOT less than 3");

    /*
    // Solution 1: linear scan
    for(int i=1; i<nums.size(); ++i)
    {
        if(nums[i] < nums[i-1])
            return i-1;
    }
    */

    int l=0;
    int r=nums.size();
    while(l < r)
    {
        int m = l + (r-l)/2;
        if(nums[m] > nums[m+1])
        {
            r = m;
        }
        else
        {
            l = m+1;
        }
    }
    return l;
}


void search_33_scaffold(string input, int target, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray_t<int>(input);
    int actual = ss.search_33(nums, target);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void search_81_scaffold(string input, int target, bool expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray_t<int>(input);
    bool actual = ss.search_81(nums, target);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findMin_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray_t<int>(input);
    int actual = ss.findMin(nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findPeakElement_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray_t<int>(input);
    int actual = ss.findPeakElement(nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void peakIndexInMountainArray_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray_t<int>(input);
    int actual = ss.peakIndexInMountainArray(nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running search_33 tests:";
    TIMER_START(search_33);
    search_33_scaffold("[1,3,5,6]", 0, -1);
    search_33_scaffold("[4,5,6,7,0,1,2]", 5, 1);
    search_33_scaffold("[4,5,6,7,0,1,2]", 1, 5);
    search_33_scaffold("[4,5,6,7,0,1,2]", 3, -1);
    TIMER_STOP(search_33);
    util::Log(logESSENTIAL) << "search_33 using " << TIMER_MSEC(search_33) << " milliseconds";

    util::Log(logESSENTIAL) << "Running search_81 tests:";
    TIMER_START(search_81);
    search_81_scaffold("[1,3,5,6]", 0, false);
    search_81_scaffold("[4,5,6,7,0,1,2]", 5, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 1, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 3, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", 3, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", 0, true);
    search_81_scaffold("[1,1,3,1]", 3, true);
    TIMER_STOP(search_81);
    util::Log(logESSENTIAL) << "search_81 using " << TIMER_MSEC(search_81) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findMin tests:";
    TIMER_START(findMin);
    findMin_scaffold("[1,3,5,6]", 1);
    findMin_scaffold("[4,5,6,7,0,1,2]", 0);
    findMin_scaffold("[2,5,6,0,0,1,2]", 0);
    findMin_scaffold("[1,1,3,1]", 1);
    TIMER_STOP(findMin);
    util::Log(logESSENTIAL) << "findMin using " << TIMER_MSEC(findMin) << " milliseconds";

    util::Log(logESSENTIAL) << "Running peakIndexInMountainArray tests:";
    TIMER_START(peakIndexInMountainArray);
    peakIndexInMountainArray_scaffold("[0,1,0]", 1);
    peakIndexInMountainArray_scaffold("[4,5,6,7,0,1]", 3);
    peakIndexInMountainArray_scaffold("[2,5,6,0,1]", 2);
    peakIndexInMountainArray_scaffold("[1,2,3,1]", 2);
    TIMER_STOP(peakIndexInMountainArray);
    util::Log(logESSENTIAL) << "peakIndexInMountainArray using " << TIMER_MSEC(peakIndexInMountainArray) << " milliseconds";
}
