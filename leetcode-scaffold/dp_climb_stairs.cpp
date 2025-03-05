#include "leetcode.h"

using namespace std;

/* leetcode: 70, 746, 1137 */

class Solution {
public:
    int climbStairs(int n);
    int minCostClimbingStairs(vector<int>& cost);
    int tribonacci(int n);
};


int Solution::climbStairs(int n) {
/*
    You are climbing a stair case. It takes n steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top? 
    Note: Given n will be a positive integer.
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

if (0) { // navie solution
    // dp[n] means ways to get n steps
    // Solution: dp[n] = dp[n-1] + dp[n-2]
    vector<int> dp(n+1, 0);
    dp[0] = 1; dp[1] = 1; // trivial cases
    for (int i=2; i<=n; ++i) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
}

{ // solution with optimization of space usage
    int dp1=1, dp2=1;
    for (int i=2; i<=n; ++i) {
        int dp3 = dp2+dp1;
        dp1 = dp2;
        dp2 = dp3;
    }
    return dp2;
}

}


int Solution::minCostClimbingStairs(vector<int>& cost) {
/*
    On a staircase, the i-th step has some non-negative cost cost[i] assigned (0-indexed). Once you pay the cost, you can either climb 1 or 2 steps.
    You need to find minimum cost to reach the top of the floor (**you have to past the last stair**), and you can either start from the step with index 0, or the step with index 1.
    Example 1:
        Input: cost = [10, 15, 20]
        Output: 15
        Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
    Example 2:
        Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
        Output: 6
        Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
*/
if (0) { // naive solution
    // dp[i] means minCost to reach i
    // Solution: dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
    int n = cost.size();
    vector<int> dp(n+1, 0);
    dp[0] = 0; dp[1] = 0; // trivial cases
    for (int i=2; i<=n; ++i) {
        dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]);
    }
    return dp[n];
}

{ // solution with optimization of space usage
    int dp1=0, dp2=0;
    int n = cost.size();
    for (int i=2; i<=n; ++i) {
        int tmp = min(dp1+cost[i-1], dp2+cost[i-2]);
        dp2 = dp1;
        dp1 = tmp;
    }
    return dp1;
}

}

int Solution::tribonacci(int n) {
/*
    The Tribonacci sequence Tn is defined as follows: T0 = 0, T1 = 1, T2 = 1, and T(n+3) = T(n) + T(n+1) + T(n+2) for n >= 0. Given n, return the value of T(n).
*/
{ // solution with optimization of space usage
    int t[4] = {0, 1, 1, 0};
    for (int i=3; i<=n; ++i) {
        t[3] = t[0] + t[1] + t[2];
        // shift t to left by one
        t[0] = t[1];
        t[1] = t[2];
        t[2] = t[3];
    }
    //return t[min(n, 3)];
    return n<3 ? t[n] : t[3];
}

}


void climbStairs_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.climbStairs(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void minCostClimbingStairs_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.minCostClimbingStairs(costs);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void tribonacci_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.tribonacci(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running climbStairs tests:");
    TIMER_START(climbStairs);
    climbStairs_scaffold(2, 2);
    climbStairs_scaffold(3, 3);
    climbStairs_scaffold(4, 5);
    TIMER_STOP(climbStairs);
    SPDLOG_WARN("climbStairs tests use {} ms", TIMER_MSEC(climbStairs));

    SPDLOG_WARN("Running minCostClimbingStairs tests:");
    TIMER_START(minCostClimbingStairs);
    minCostClimbingStairs_scaffold("[10, 15, 20]", 15);
    minCostClimbingStairs_scaffold("[1, 100, 1, 1, 1, 100, 1, 1, 100, 1]", 6);
    TIMER_STOP(minCostClimbingStairs);
    SPDLOG_WARN("minCostClimbingStairs tests use {} ms", TIMER_MSEC(minCostClimbingStairs));

    SPDLOG_WARN("Running tribonacci tests:");
    TIMER_START(tribonacci);
    tribonacci_scaffold(2, 1);
    tribonacci_scaffold(3, 2);
    tribonacci_scaffold(4, 4);
    tribonacci_scaffold(25, 1389537);
    TIMER_STOP(tribonacci);
    SPDLOG_WARN("tribonacci tests use {} ms", TIMER_MSEC(tribonacci));
}
