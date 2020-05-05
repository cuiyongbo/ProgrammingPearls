#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 11, 42 */

class Solution 
{
public:
    int maxArea(const vector<int>& height);
    int trap(vector<int>& height);
};

int Solution::maxArea(const vector<int>& height)
{
    /*
        Given n non-negative integers a1, a2, …, an, where each represents a point at coordinate (i, ai). 
        n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
        Find two lines, which together with x-axis forms a container, such that the container contains the most water.
        Note: You may not slant the container and n is at least 2.

        For example, Given input: [1 3 2 4], output: 6
        explanation: use 3, 4, we have the following, which contains (4th-2nd) * min(3, 4) = 2 * 3 = 6 unit of water.

            1    |
            2 |~~|
            3 |~~| 
            4 |~~|
    */

    int ans = 0;
    int l=0, r=(int)height.size() - 1;
    while(l != r)
    {
        ans = std::max(ans, std::min(height[l], height[r])*(r-l));

        // move toward the direction that may maxify the answer
        (height[l] < height[r]) ? (++l) : (--r);
    }
    return ans;
}

int Solution::trap(vector<int>& height)
{
    /*
        Given n non-negative integers representing an elevation map where the width of each bar is 1, 
        compute how much water it is able to trap after raining.
    */

    int ans = 0;
    int size = (int)height.size();
    int l=0, max_l = height[l];
    int r=size-1, max_r = height[r];
    while(l != r)
    {
        if(max_l < max_r)
        {
            ans += max_l - height[l];
            max_l = std::max(max_l, height[++l]);
        }
        else
        {
            ans += max_r - height[r];
            max_r = std::max(max_r, height[--r]);
        }
    }
    return ans;
}

void maxArea_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input);
    int actual = ss.maxArea(g);
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

void trap_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input);
    int actual = ss.trap(g);
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

    util::Log(logESSENTIAL) << "Running maxArea tests: ";
    TIMER_START(maxArea);
    maxArea_scaffold("[1,3,2,4]", 6);
    TIMER_STOP(maxArea);
    util::Log(logESSENTIAL) << "maxArea using " << TIMER_MSEC(maxArea) << " milliseconds";

    util::Log(logESSENTIAL) << "Running trap tests: ";
    TIMER_START(trap);
    trap_scaffold("[1,3,2,4]", 1);
    trap_scaffold("[0,1,0,2,1,0,1,3,2,1,2,1]", 6);
    TIMER_STOP(trap);
    util::Log(logESSENTIAL) << "trap using " << TIMER_MSEC(trap) << " milliseconds";

}
