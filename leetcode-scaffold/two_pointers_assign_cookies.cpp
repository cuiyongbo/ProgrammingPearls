#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 455, 209 */

class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s);
    int minSubArrayLen(int s, vector<int>& nums);
};

int Solution::findContentChildren(vector<int>& g, vector<int>& s) {
/*
    Assume you are an awesome parent and want to give your children some cookies. 
    But you should give each child at most one cookie. Each child i has a greed factor g_i, 
    which is the minimum size of a cookie that the child will be content with; 
    and each cookie j has a size s_j. If s_j >= g_i, we can assign the cookie j to the child i, 
    and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

    Note:
        You may assume the greed factor is always positive.
        You cannot assign more than one cookie to one child.
*/
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int child_cnt = g.size();
    int cookie_cnt = s.size();
    int j = 0;
    for (int i=0; i<cookie_cnt && j<child_cnt; ++i) {
        if (g[j]>s[i]) {
            continue;
        }
        ++j;
    }
    return j;
}

int Solution::minSubArrayLen(int s, vector<int>& nums) {
/*
    Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray whose sum ≥ s. 
    If there isn’t one, return 0 instead.
*/

{
    int ans = INT32_MAX;
    int sum = 0;
    int sz = nums.size();
    for (int i=0, j=0; i<sz;) {
        while (j<sz&&sum<s) {
            sum += nums[j++];
        }
        if (sum < s) {
            break;
        }
        ans = min(ans, j-i);
        sum -= nums[i++];
    }
    return ans==INT32_MAX ? 0 : ans;
}

void findContentChildren_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    vector<int> s = stringTo1DArray<int>(input2);
    int actual = ss.findContentChildren(g, s);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void minSubArrayLen_scaffold(int input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input2);
    int actual = ss.minSubArrayLen(input1, nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running findContentChildren tests: ";
    TIMER_START(findContentChildren);
    findContentChildren_scaffold("[1,2,3]", "[1,1]", 1);
    findContentChildren_scaffold("[1,2]", "[1,2,3]", 2);
    TIMER_STOP(findContentChildren);
    util::Log(logESSENTIAL) << "findContentChildren using " << TIMER_MSEC(findContentChildren) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minSubArrayLen tests: ";
    TIMER_START(minSubArrayLen);
    minSubArrayLen_scaffold(7, "[2,3,1,2,4,3]", 2);
    TIMER_STOP(minSubArrayLen);
    util::Log(logESSENTIAL) << "minSubArrayLen using " << TIMER_MSEC(minSubArrayLen) << " milliseconds";
}