#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 455, 209 */

class Solution 
{
public:
    int findContentChildren(vector<int>& g, vector<int>& s);
    int minSubArrayLen(int s, vector<int>& nums);
};

int Solution::findContentChildren(vector<int>& g, vector<int>& s)
{
    /*
        Assume you are an awesome parent and want to give your children some cookies. 
        But, you should give each child at most one cookie. Each child i has a greed factor gi, 
        which is the minimum size of a cookie that the child will be content with; 
        and each cookie j has a size sj. If sj >= gi, we can assign the cookie j to the child i, 
        and the child i will be content. Your goal is to maximize the number of your content children 
        and output the maximum number.

        Note:
            You may assume the greed factor is always positive.
            You cannot assign more than one cookie to one child.
    */

    std::sort(g.begin(), g.end());
    std::sort(s.begin(), s.end());

    int count = 0;
    int child_count = (int)g.size();
    int cookie_count = (int)s.size();
    for(int i=0, j=0; i<child_count && j<cookie_count;)
    {
        while(j < cookie_count && g[i] > s[j])
        {
            ++j;
        }

        if(j != cookie_count)
        {
            count++;
            i++; j++;
        }
    }
    return count;
}

int Solution::minSubArrayLen(int s, vector<int>& nums)
{
    /*
        Given an array of n positive integers and a positive integer s, 
        find the minimal length of a contiguoussubarray of which the sum ≥ s. 
        If there isn’t one, return 0 instead.
    */

    int sum = 0;
    int ans = INT32_MAX;
    int size = (int)nums.size();
    for(int i=0, j=0; i<size;)
    {
        while(j < size && sum < s)
        {
            sum += nums[j++];
        }

        if(sum < s) break;
        ans = std::min(ans, j-i);
        sum -= nums[i++];
    }

    return ans == INT32_MAX ? 0 : ans;
}

void findContentChildren_scaffold(string input1, string input2, int expectedResult)
{
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    vector<int> s = stringTo1DArray<int>(input2);
    int actual = ss.findContentChildren(g, s);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void minSubArrayLen_scaffold(int input1, string input2, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input2);
    int actual = ss.minSubArrayLen(input1, nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
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