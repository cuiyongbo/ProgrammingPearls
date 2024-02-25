#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 167, 15, 16 */

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target);
    vector<vector<int>> threeSum(vector<int>& nums);
    int threeSumClosest(vector<int> &num, int target);
};

vector<int> Solution::twoSum(vector<int>& nums, int target) {
/*
    Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

    The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.
    Please note that your returned answers (both index1 and index2) are one-based. You may assume that each input would have exactly one solution and you may not use the same element twice.

    Given an input: numbers={2, 7, 11, 15}, target=9, Output: index1=1, index2=2
*/
    int l=0;
    int r=nums.size()-1;
    while (l != r) {
        int m = nums[l]+nums[r];
        if (m == target) {
            break;
        } else if (m < target) {
            ++l;
        } else {
            --r;
        }
    }
    return {l+1, r+1};
}

vector<vector<int>> Solution::threeSum(vector<int>& nums) {
/*
    Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
    Find all unique triplets in the array which gives the sum of zero. Note: The solution set must not contain duplicate triplets.
    For example, given array S = [-1, 0, 1, 2, -1, -4], A solution set is:
        [
            [-1, 0, 1],
            [-1, -1, 2]
        ]
*/
    sort(nums.begin(), nums.end());
    //print_vector(nums);

    vector<vector<int>> ans;
    int sz = nums.size();
    for (int i=0; i<sz-2; ++i) {
        if (nums[i] > 0) { // prune useless branches
            break;
        }
        if (i>0 && nums[i] == nums[i-1]) { // skip repeated candidate(s)
            continue;
        }
        int l = i+1;
        int r = sz-1;
        while (l < r) {
            int m = nums[i] + nums[l] + nums[r];
            //printf("i=%d, l=%d, r=%d, m=%d\n", i, l, r, m);
            if (m == 0) {
                ans.push_back({nums[i], nums[l++], nums[r--]});
                while (l<r && nums[l] == nums[l-1]) {
                    ++l;
                }
                while (l<r && nums[r] == nums[r+1]) {
                    --r;
                }
            } else if (m < 0) {
                ++l;
            } else {
                --r;
            }
        }
    }
    return ans;
}

int Solution::threeSumClosest(vector<int>& nums, int target) {
/*
    Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. 
    Return the sum of the three integers. You may assume that each input would have exactly one solution.
    Example: Given array nums = [-1, 2, 1, -4], and target = 1. The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
*/
    sort(nums.begin(), nums.end());
    int ans = 0;
    int diff = INT32_MAX;
    int sz = nums.size();
    for (int i=0; i<sz-2; ++i) {
        int l = i+1;
        int r = sz-1;
        while (l < r) {
            int m = nums[i] + nums[l] + nums[r];
            if (diff > abs(m-target)) {
                diff = abs(m-target);
                ans = m;
            }
            if (m == target) {
                return target;
            } else if (m < target) {
                ++l;
            } else {
                --r;
            }
        }
    }
    return ans;
}

void twoSum_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.twoSum(A, input2);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << numberVectorToString<int>(actual);
    }
}

void threeSum_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.threeSum(A);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& s: actual) {
            util::Log(logERROR) << numberVectorToString<int>(s);
        }
    }
}

void threeSumClosest_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.threeSumClosest(A, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running twoSum tests: ";
    TIMER_START(twoSum);
    twoSum_scaffold("[1,2]", 3, "[1,2]");
    twoSum_scaffold("[2, 7, 11, 15]", 9, "[1,2]");
    TIMER_STOP(twoSum);
    util::Log(logESSENTIAL) << "twoSum using " << TIMER_MSEC(twoSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running threeSum tests: ";
    TIMER_START(threeSum);
    threeSum_scaffold("[-1,0,1,2,-1,-4]", "[[-1,-1,2],[-1,0,1]]");
    TIMER_STOP(threeSum);
    util::Log(logESSENTIAL) << "threeSum using " << TIMER_MSEC(threeSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running threeSumClosest tests: ";
    TIMER_START(threeSumClosest);
    threeSumClosest_scaffold("[-1, 2, 1, -4]", 1, 2);
    TIMER_STOP(threeSumClosest);
    util::Log(logESSENTIAL) << "threeSumClosest using " << TIMER_MSEC(threeSumClosest) << " milliseconds";
}
