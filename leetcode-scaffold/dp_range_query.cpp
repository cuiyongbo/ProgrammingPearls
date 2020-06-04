#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 303, 1218 */

/*
    Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

    Example:
    Given nums = [-2, 0, 3, -5, 2, -1]
    sumRange(0, 2) -> 1
    sumRange(2, 5) -> -1
    sumRange(0, 5) -> -3
    Note:
        You may assume that the array does not change.
        There are many calls to sumRange function.
*/

class NumArray
{
public:
    NumArray(const vector<int>& nums)
    {
        m_sum.resize(nums.size() + 1, 0);
        std::partial_sum(nums.begin(), nums.end(), std::next(m_sum.begin()));
    }

    int sumRange(int i, int j) { return m_sum[j+1] - m_sum[i];}

private:
    vector<int> m_sum;
};

class Solution 
{
public:
  int longestSubsequence(vector<int>& arr, int d);
};

int Solution::longestSubsequence(vector<int>& arr, int d)
{
    /*
        Given an integer array arr and an integer difference, 
        return the length of the longest subsequence in arr 
        which is an arithmetic sequence such that the difference 
        between adjacent elements in the subsequence equals difference.

        Example 1:
        Input: arr = [1,2,3,4], difference = 1
        Output: 4
        Explanation: The longest arithmetic subsequence is [1,2,3,4].

        Example 2:
        Input: arr = [1,3,5,7], difference = 1
        Output: 1
        Explanation: The longest arithmetic subsequence is any single element.

        Example 3:
        Input: arr = [1,5,7,8,5,3,4,2,1], difference = -2
        Output: 4
        Explanation: The longest arithmetic subsequence is [7,5,3,1].
    */

    // dp[x] means longestSubsequence ending with x
    // dp[x] = max(0, dp[x-diff]) + 1

    int ans = 0;
    unordered_map<int, int> dp;
    for(const auto x: arr)
    {
        ans = std::max(ans, dp[x]=dp[x-d]+1);        
    }
    return ans;
}

void longestSubsequence_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.longestSubsequence(nums, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << "expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << "expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running NumArray tests:";
    TIMER_START(NumArray);

    NumArray na({-2, 0, 3, -5, 2, -1});
    assert(na.sumRange(0,2) == 1);
    assert(na.sumRange(2,5) == -1);
    assert(na.sumRange(0,5) == -3);
    assert(na.sumRange(0,0) == -2);

    TIMER_STOP(NumArray);
    util::Log(logESSENTIAL) << "NumArray using " << TIMER_MSEC(NumArray) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestSubsequence tests:";
    TIMER_START(longestSubsequence);
    longestSubsequence_scaffold("[1,2,3,4]", 1, 4);
    longestSubsequence_scaffold("[1,3,5,7]", 1, 1);
    longestSubsequence_scaffold("[1,5,7,8,5,3,4,2,1]", -2, 4);
    TIMER_STOP(longestSubsequence);
    util::Log(logESSENTIAL) << "longestSubsequence using " << TIMER_MSEC(longestSubsequence) << " milliseconds";
}
