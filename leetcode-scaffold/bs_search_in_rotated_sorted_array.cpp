#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 33, 81, 153, 154, 162, 852 */
class Solution {
public:
    int search_33(std::vector<int>& nums, int target);
    bool search_81(std::vector<int>& nums, int target);
    int findMin(std::vector<int> &num);
    int findPeakElement(std::vector<int>& nums);
    int peakIndexInMountainArray(std::vector<int>& A);
};


/*
    Suppose an array sorted in ascending order is rotated at some pivot unknown to you. (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).
    You are given a target value to search. If found in the array return its index, otherwise return -1.
    You may assume no duplicate exists in the array. Your algorithm's runtime complexity must be in the order of O(log n).
*/
int Solution::search_33(std::vector<int>& nums, int target) {
    std::function<int(int, int)> dac = [&] (int l, int r) {
        if (l > r) { // trivial case
            return -1;
        }

        // l must be less or equal than r
        int m = (l+r)/2;
        if (nums[m] == target) {
            return m;
        }

        if (nums[l] < nums[r]) { // nums[l:r] is sorted
            // perform normal binary search
            if (nums[m] < target) {
                l = m+1;
            } else { 
                r = m-1;
            }
            return dac(l, r);
        } else {
            int i = dac(l, m-1);
            if (i == -1) {
                i = dac(m+1, r);
            }
            return i;
        }
    };
    return dac(0, nums.size()-1);
}


/*
    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
    (i.e., [0,0,1,2,2,5,6] might become [2,5,6,0,0,1,2]). You are given a target value to search. 
    If found in the array return true, otherwise return false.
    The array may contain duplicates. Your algorithm's runtime complexity must be in the order of O(log n).
*/
bool Solution::search_81(std::vector<int>& nums, int target) {

    std::function<bool(int, int)> dac = [&] (int l, int r) {
        if (l > r) { // trivial case
            return false;
        }

        // l must be less or equal than r
        int m = (l+r)/2;
        if (nums[m] == target) {
            return true;
        }

        if (nums[l] < nums[r]) { // nums[l:r] is sorted
            // perform normal binary search
            if (nums[m] < target) {
                l = m+1;
            } else { 
                r = m-1;
            }
            return dac(l, r);
        } else {
            return dac(l, m-1) || dac(m+1, r);
        }
    };
    return dac(0, nums.size()-1);
}


/*
    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2). Find the minimum element. You may assume no duplicate exists in the array.
*/
int Solution::findMin(std::vector<int>& nums) {
    std::function<int(int, int)> dac = [&] (int l, int r) {
        if (l==r) { // trivial case
            return nums[l];
        }
        if (nums[l] < nums[r]) { // nums[l:r] is sorted
            return nums[l];
        } else {
            int m = (l+r)/2;
            return std::min(dac(l, m), dac(m+1, r));
        }
    };
    return dac(0, nums.size()-1);
}


/*
    A peak element is an element that is greater than its neighbors.
    Given an input array nums, where nums[i] != nums[i+1], find a peak element and return its index.
    The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
    Note:
        You may imagine that nums[-1] = nums[n] = -inf.
        You must write an algorithm that runs in O(log n) time.
    Example 1:
        Input: nums = [1,2,3,1]
        Output: 2
    Example 2:
        Input: nums = [1,2,1,3,5,6,4]
        Output: 1 or 5 
*/
int Solution::findPeakElement(std::vector<int>& nums) {

{ // naive solution
    int sz = nums.size();
    function<int(int, int)> dac = [&] (int l, int r) {
        if (l>r) { // trivial case
            return -1;
        }
        int m = (l+r)/2;
        //int left = (m==0) ? INT32_MIN : nums[m-1];
        //int right = (m==sz-1) ? INT32_MIN : nums[m+1];
        //if (nums[m]>left && nums[m]>right) {
        if ((m==0||nums[m]>nums[m-1]) && (m==sz-1||nums[m]>nums[m+1])) {
            return m;
        } else {
            int i = dac(l, m-1);
            if (i == -1) {
                i = dac(m+1, r);
            }
            return i;
        }
    };
    return dac(0, sz-1);
}

{ // trick version
    int sz = nums.size();
    std::vector<int> twin(sz+2, INT32_MIN);
    std::copy(nums.begin(), nums.end(), twin.begin()+1);
    function<int(int, int)> dac = [&] (int l, int r) {
        if (l>r) { // trivial case
            return -1;
        }
        int m = (l+r)/2;
        if (twin[m]>twin[m-1] && twin[m]>twin[m+1]) {
            return m;
        } else {
            int i = dac(l, m-1);
            if (i==-1) {
                i = dac(m+1, r);
            }
            return i;
        }
    };
    return dac(1, sz)-1;
}

{ // another trick version
    int sz = nums.size();
    // trival cases
    if (sz == 1) {
        return 0;
    } else {
        if (nums[0] > nums[1]) {
            return 0;
        }
        if (nums[sz-2] < nums[sz-1]) {
            return sz-1;
        }
    }
    std::function<int(int, int)> dac = [&] (int l, int r) {
        if (l>r) { //trivial case
            return -1;
        }
        int m = (l+r)/2;
        if (nums[m-1]<nums[m] && nums[m]>nums[m+1]) {
            return m;
        }
        int ans = dac(l, m-1);
        if (ans == -1) {
            ans = dac(m+1, r);
        }
        return ans;
    };
    return dac(1, sz-2);
}

}


/*
    Let’s call an array A a mountain if the following properties hold:
        A.length >= 3
        There exists some **0 < i < A.length-1** such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length-1]
    Given an array that is definitely a mountain, return such index.
*/
int Solution::peakIndexInMountainArray(std::vector<int>& nums) {

if (0) { // iterative version solution
    int l = 0;
    int r = nums.size();
    while (l < r) {
        int m = (l+r)/2;
        if (nums[m]<nums[m+1]) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}

{ // recursive version solution
    int sz = nums.size();
    std::function<int(int, int)> dac = [&] (int l, int r) {
        if (l>r) { //trivial case
            return -1;
        }
        int m = (l+r)/2;
        if (nums[m-1]<nums[m] && nums[m]>nums[m+1]) {
            return m;
        }
        int ans = dac(l, m-1);
        if (ans == -1) {
            ans = dac(m+1, r);
        }
        return ans;
    };
    return dac(1, sz-2);
}

}

void search_33_scaffold(std::string input, int target, int expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.search_33(nums, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, Actual: " << actual;
    }
}

void search_81_scaffold(std::string input, int target, bool expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    bool actual = ss.search_81(nums, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, Actual: " << actual;
    }
}

void findMin_scaffold(std::string input, int expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.findMin(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findPeakElement_scaffold(std::string input, std::string expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    std::vector<int> expected = stringTo1DArray<int>(expectedResult);
    int actual = ss.findPeakElement(nums);
    if (std::any_of(expected.begin(), expected.end(), [&](int n) {return n == actual;})) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void peakIndexInMountainArray_scaffold(std::string input, int expectedResult)
{
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.peakIndexInMountainArray(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running search_33 tests:";
    TIMER_START(search_33);
    search_33_scaffold("[1,3,5,6]", 0, -1);
    search_33_scaffold("[1,3,5,6]", 1, 0);
    search_33_scaffold("[1,3,5,6]", 2, -1);
    search_33_scaffold("[1,3,5,6]", 3, 1);
    search_33_scaffold("[1,3,5,6]", 4, -1);
    search_33_scaffold("[1,3,5,6]", 5, 2);
    search_33_scaffold("[1,3,5,6]", 6, 3);
    search_33_scaffold("[4,5,6,7,0,1,2]", 0, 4);
    search_33_scaffold("[4,5,6,7,0,1,2]", 1, 5);
    search_33_scaffold("[4,5,6,7,0,1,2]", 2, 6);
    search_33_scaffold("[4,5,6,7,0,1,2]", 3, -1);
    search_33_scaffold("[4,5,6,7,0,1,2]", 4, 0);
    search_33_scaffold("[4,5,6,7,0,1,2]", 5, 1);
    search_33_scaffold("[4,5,6,7,0,1,2]", 6, 2);
    search_33_scaffold("[4,5,6,7,0,1,2]", 7, 3);
    search_33_scaffold("[4,5,6,7,0,1,2]", 8, -1);
    search_33_scaffold("[4,5,6,7,0,1,2]", -1, -1);
    TIMER_STOP(search_33);
    util::Log(logESSENTIAL) << "search_33 using " << TIMER_MSEC(search_33) << " milliseconds";

    util::Log(logESSENTIAL) << "Running search_81 tests:";
    TIMER_START(search_81);
    search_81_scaffold("[1,3,5,6]", 0, false);
    search_81_scaffold("[1,3,5,6]", 1, true);
    search_81_scaffold("[1,3,5,6]", 2, false);
    search_81_scaffold("[1,3,5,6]", 3, true);
    search_81_scaffold("[1,3,5,6]", 4, false);
    search_81_scaffold("[1,3,5,6]", 5, true);
    search_81_scaffold("[1,3,5,6]", 6, true);
    search_81_scaffold("[1,3,5,6]", 7, false);
    search_81_scaffold("[4,5,6,7,0,1,2]", -1, false);
    search_81_scaffold("[4,5,6,7,0,1,2]", 0, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 1, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 2, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 3, false);
    search_81_scaffold("[4,5,6,7,0,1,2]", 4, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 5, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 6, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 7, true);
    search_81_scaffold("[4,5,6,7,0,1,2]", 8, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", -1, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", 0, true);
    search_81_scaffold("[2,5,6,0,0,1,2]", 1, true);
    search_81_scaffold("[2,5,6,0,0,1,2]", 2, true);
    search_81_scaffold("[2,5,6,0,0,1,2]", 3, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", 4, false);
    search_81_scaffold("[2,5,6,0,0,1,2]", 5, true);
    search_81_scaffold("[2,5,6,0,0,1,2]", 6, true);
    search_81_scaffold("[2,5,6,0,0,1,2]", 7, false);
    search_81_scaffold("[1,1,3,1]", 3, true);
    search_81_scaffold("[1,0,1,1,1]", 0, true);
    TIMER_STOP(search_81);
    util::Log(logESSENTIAL) << "search_81 using " << TIMER_MSEC(search_81) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findMin tests:";
    TIMER_START(findMin);
    findMin_scaffold("[1,3,5,6]", 1);
    findMin_scaffold("[1,3,5]", 1);
    findMin_scaffold("[4,5,6,7,0,1,2]", 0);
    findMin_scaffold("[2,5,6,0,0,1,2]", 0);
    findMin_scaffold("[1,1,3,1]", 1);
    findMin_scaffold("[2,2,2,0,1]", 0);
    TIMER_STOP(findMin);
    util::Log(logESSENTIAL) << "findMin using " << TIMER_MSEC(findMin) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findPeakElement tests:";
    TIMER_START(findPeakElement);
    findPeakElement_scaffold("[2,3]", "[1]");
    findPeakElement_scaffold("[1]", "[0]");
    findPeakElement_scaffold("[1,2,3,1]", "[2]");
    findPeakElement_scaffold("[1,2,1,3,5,6,4]", "[1,5]");
    TIMER_STOP(findPeakElement);
    util::Log(logESSENTIAL) << "findPeakElement using " << TIMER_MSEC(findPeakElement) << " milliseconds";

    util::Log(logESSENTIAL) << "Running peakIndexInMountainArray tests:";
    TIMER_START(peakIndexInMountainArray);
    peakIndexInMountainArray_scaffold("[0,1,0]", 1);
    peakIndexInMountainArray_scaffold("[4,5,6,7,0,1]", 3);
    peakIndexInMountainArray_scaffold("[2,5,6,0,1]", 2);
    peakIndexInMountainArray_scaffold("[1,2,3,1]", 2);
    peakIndexInMountainArray_scaffold("[2,1,3,1]", 2);
    peakIndexInMountainArray_scaffold("[0,2,1,0]", 1);
    peakIndexInMountainArray_scaffold("[0,10,5,5,2]", 1);
    peakIndexInMountainArray_scaffold("[0,5,10,5,2]", 2);
    peakIndexInMountainArray_scaffold("[0,5,5,10,2]", 3);
    TIMER_STOP(peakIndexInMountainArray);
    util::Log(logESSENTIAL) << "peakIndexInMountainArray using " << TIMER_MSEC(peakIndexInMountainArray) << " milliseconds";
}
