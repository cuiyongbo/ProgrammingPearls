#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 70, 746, 1137 */

class Solution 
{
public:
    int climbStairs(int n);
    int minCostClimbingStairs(vector<int>& cost);
    int tribonacci(int n);
};

int Solution::climbStairs(int n)
{
    /*
        You are climbing a stair case. It takes n steps to reach to the top.
        Each time you can either climb 1 or 2 steps. In how many distinct ways 
        can you climb to the top? Note: Given n will be a positive integer.

            Example 1:
            Input: 2
            Output: 2
            Explanation: There are two ways to climb to the top.
            1. 1 step + 1 step
            2. 2 steps

            Example 2:
            Input: 3
            Output: 3
            Explanation: There are three ways to climb to the top.
            1. 1 step + 1 step + 1 step
            2. 1 step + 2 steps
            3. 2 steps + 1 step
    */

    // dp[n] means ways to get n steps
    // dp[n] = dp[n-1] + dp[n-2]

    vector<int> dp(n+1, 0);
    dp[0] = 1; dp[1] = 1;
    for(int i=2; i<=n; i++)
        dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}

int Solution::minCostClimbingStairs(vector<int>& cost)
{
    /*
        On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).
        Once you pay the cost, you can either climb 1 or 2 steps. You need to find minimum 
        cost to reach the top of the floor, and you can either start from the step with index 0, 
        or the step with index 1.

        Example 1:
        Input: cost = [10, 15, 20]
        Output: 15
        Explanation: Cheapest is start on cost[1], pay that cost and go to the top.

        Example 2:
        Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
        Output: 6
        Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
    */

    int n = (int)cost.size();

    // dp[i] means minCost to reach i
    // dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])

/*
    vector<int> dp(n+1, INT32_MAX);
    dp[0] = 0; dp[1] = 0;
    for(int i=2; i<n+1; i++)
        dp[i] = std::min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]);
    return dp[n];
*/

    // O(1) space
    int dp1=0, dp2=0, dp=0;
    for(int i=2; i<n+1; i++)
    {
        dp = std::min(dp1+cost[i-1], dp2+cost[i-2]);
        dp2 = dp1;
        dp1 = dp;
    }
    return dp;
}

int Solution::tribonacci(int n)
{
    /*
        The Tribonacci sequence Tn is defined as follows: 
        T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
        Given n, return the value of Tn.

        Example 1:
        Input: n = 4
        Output: 4
        Explanation:
        T_3 = 0 + 1 + 1 = 2
        T_4 = 1 + 1 + 2 = 4

        Example 2:
        Input: n = 25
        Output: 1389537
    */

    int t[4] = {0, 1, 1, 0};
    if(n<3) return t[n];

    for(int i=3; i<=n; i++)
    {
        t[3] = t[0] + t[1] + t[2];
        t[0] = t[1];
        t[1] = t[2];
        t[2] = t[3];
    }
    return t[3];
}

void climbStairs_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.climbStairs(input);
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

void minCostClimbingStairs_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.minCostClimbingStairs(costs);
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

void tribonacci_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.tribonacci(input);
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

    util::Log(logESSENTIAL) << "Running climbStairs tests:";
    TIMER_START(climbStairs);
    climbStairs_scaffold(2, 2);
    climbStairs_scaffold(3, 3);
    TIMER_STOP(climbStairs);
    util::Log(logESSENTIAL) << "climbStairs using " << TIMER_MSEC(climbStairs) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minCostClimbingStairs tests:";
    TIMER_START(minCostClimbingStairs);
    minCostClimbingStairs_scaffold("[10, 15, 20]", 15);
    minCostClimbingStairs_scaffold("[1, 100, 1, 1, 1, 100, 1, 1, 100, 1]", 6);
    TIMER_STOP(minCostClimbingStairs);
    util::Log(logESSENTIAL) << "minCostClimbingStairs using " << TIMER_MSEC(minCostClimbingStairs) << " milliseconds";

    util::Log(logESSENTIAL) << "Running tribonacci tests:";
    TIMER_START(tribonacci);
    tribonacci_scaffold(2, 1);
    tribonacci_scaffold(3, 2);
    tribonacci_scaffold(4, 4);
    tribonacci_scaffold(25, 1389537);
    TIMER_STOP(tribonacci);
    util::Log(logESSENTIAL) << "tribonacci using " << TIMER_MSEC(tribonacci) << " milliseconds";
}
