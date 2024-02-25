#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 198, 213, 309, 740, 790, 801 */

class Solution {
public:
    int rob_198(vector<int>& moneys);
    int rob_213(vector<int>& moneys);
    int maxProfit(vector<int>& prices);
    int deleteAndEarn(vector<int>& nums);
    int numTilings(int N);
    int minSwap(vector<int>& A, vector<int>& B);
};

int Solution::rob_198(vector<int>& moneys) {
/*
    You are a professional robber planning to rob houses along a street. Each house has a certain amount of money, the only constraint stopping you from robbing them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
*/
    // dp[i] means the max profit when robbing house[:i], i is inclusive
    // dp[i] = max(dp[i-2]+moneys[i], dp[i-1])

    int sz = moneys.size();
    vector<int> dp = moneys;
    dp[0] = moneys[0]; dp[1] = max(moneys[0], moneys[1]);
    for (int i=2; i<sz; ++i) {
        dp[i] = max(dp[i-1], dp[i-2]+moneys[i]);
    }
    return dp[sz-1];
}

int Solution::rob_213(vector<int>& moneys) {
/*
    You are a professional robber planning to rob houses along a street. Each house has a certain amount of money. 
    **All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one.** 
    Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
    Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
    Hint: similar to rob_198, except that we may not steal the first house and the last room at the same time
*/
    int n = moneys.size();
    // trivial cases
    if (n == 1) {
        return moneys[0];
    } else if (n == 2) {
        return max(moneys[0], moneys[1]);
    }
    auto worker = [&] (int l, int r) {
        vector<int> dp = moneys;
        dp[l] = moneys[l]; dp[l+1] = max(moneys[l], moneys[l+1]);
        for (int i=l+2; i<r; ++i) {
            dp[i] = max(dp[i-2]+moneys[i], dp[i-1]);
        }
        return dp[r-1];        
    };
    return max(worker(0, n-1), // steal hourses[0, n-1)
                worker(1, n)); // steal hourses[1, n)
}

int Solution::maxProfit(vector<int>& prices) {
/*
    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions 
    as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

        You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
        After you sell your stock, you cannot buy stock on next day. (i.e., cooldown 1 day)
*/

    // sold[i] means maxProfit when sold 
    // hold[i] = max(hold[i-1], rest[i-1] - prices[i])
    // sold[i] = hold[i-1] + prices[i]
    // rest[i] = max(rest[i-1], sold[i-1])
    // init: rest[0]=sold[0]=0, hold[0]=-inf

    int sold = 0;
    int rest = 0;
    int hold = -prices[0];
    for (auto p: prices) {
        int ps=sold, pr=rest, ph=hold;
        sold = ph + p;
        hold = max(ph, pr-p);
        rest = max(pr, ps);
    }
    return max(sold, rest);
}

int Solution::deleteAndEarn(vector<int>& nums) {
/*
    Given an array nums of integers, you can perform operations on the array. In each operation, you pick any nums[i] and delete it to earn nums[i] points. 
    After, you must delete every element equal to nums[i]-1 or nums[i]+1. You start with 0 points. Return the maximum number of points you can earn by applying such operations.
    Example 1:
        Input: nums = [3, 4, 2]
        Output: 6
        Explanation: Delete 4 to earn 4 points, consequently 3 is also deleted. Then delete 2 to earn 2 points. 6 total points are earned.
    Example 2:
        Input: nums = [2, 2, 3, 3, 3, 4]
        Output: 9
        Explanation: Delete 3 to earn 3 points, deleting both 2's and the 4. Then, delete 3 again to earn 3 points, and 3 again to earn 3 points. 9 total points are earned. 
*/
    if (nums.empty()) {
        return 0;
    }
    auto range = std::minmax_element(nums.begin(), nums.end());
    int l = *(range.first);
    int r = *(range.second);
    vector<int> points(r-l+1, 0);
    for (auto n: nums) {
        points[n-l] += n;
    }
    return rob_198(points);
}

int Solution::numTilings(int N) {
/*
    We have two types of tiles: a 2×1 domino shape, and an “L” tromino shape. These shapes may be rotated.
        XX  <- domino
        XX  <- "L" tromino
        X
    Given N, how many ways are there to tile a 2 x N board? Return your answer modulo 10^9 + 7.

    In a tiling, every square must be covered by a tile. 
    Two tilings are different if and only if there are two 4-directionally 
    adjacent cells on the board such that exactly one of the tilings has both 
    squares occupied by a tile.
*/
    // const int kMod = 1e09 + 7;
    return 0;
}

int Solution::minSwap(vector<int>& A, vector<int>& B) {
/*
    We have two integer sequences A and B of the same non-zero length. We are allowed to swap elements A[i] and B[i]. 
    Note that both elements are in the same index position in their respective sequences.
    Given A and B, return the minimum number of swaps to make both sequences strictly increasing.
    It is guaranteed that the given input always makes it possible.
*/
    // use exchange[i]/keep[i] to denote the minSwap to make A[0:i] 
    // and B[0:i] strictly increasing with/without swap(A[i], B[i])
    int n = A.size();
    vector<int> keep(n, INT32_MAX);
    vector<int> exchange(n, INT32_MAX);
    exchange[0] = 1; keep[0] = 0; // if both arrays have only one element, it doesn't matter whether we swap them or not
    for (int i=1; i<n; i++) {
        // 1. we may swap either at both positions i-1 and i or neigher
        if (A[i-1] < A[i] && B[i-1] < B[i]) { 
            keep[i] = keep[i-1]; // no need to swap
            exchange[i] = exchange[i-1]+1; // swap at both postitions i-1 and i
        }
        // 2. we may swap at position either i-1 or i
        if (A[i-1]<B[i] && B[i-1]<A[i]) {
            exchange[i] = min(exchange[i], keep[i-1]+1); // swap only at position i
            keep[i] = min(keep[i], exchange[i-1]); // swap only at position i-1
        }
    }
    return min(exchange[n-1], keep[n-1]);
}

void rob_scaffold(string input, int expectedResult, int func_no) {
    Solution ss;
    vector<int> moneys = stringTo1DArray<int>(input);
    int actual = 0;
    if (func_no == 198) {
        actual = ss.rob_198(moneys);
    } else if (func_no == 213) {
        actual = ss.rob_213(moneys);
    } else {
        util::Log(logERROR) << "func_no can only be values in [198, 213]";
        return;
    }
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ", func_no:" << func_no << ") passed";
    } else {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ", func_no:" << func_no << ") failed, actual: " << actual;
    }
}

void maxProfit_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> moneys = stringTo1DArray<int>(input);
    int actual = ss.maxProfit(moneys);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void deleteAndEarn_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> moneys = stringTo1DArray<int>(input);
    int actual = ss.deleteAndEarn(moneys);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void minSwap_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    vector<int> B = stringTo1DArray<int>(input2);
    int actual = ss.minSwap(A, B);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running rob tests:";
    TIMER_START(rob);
    rob_scaffold("[1]", 1, 198);
    rob_scaffold("[1,2,3,4,5]", 9, 198);
    rob_scaffold("[1,2,3,4]", 6, 198);
    rob_scaffold("[1,3,1,3,100]", 103, 198);
    rob_scaffold("[1,2,3,4,5]", 8, 213);
    rob_scaffold("[1,2,3,4]", 6, 213);
    rob_scaffold("[2,3,2]", 3, 213);
    rob_scaffold("[1,3,1,3,100]", 103, 213);
    rob_scaffold("[1,2,3,1]", 4, 213);
    rob_scaffold("[1]", 1, 213);
    rob_scaffold("[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]", 0, 213);
    TIMER_STOP(rob);
    util::Log(logESSENTIAL) << "rob using " << TIMER_MSEC(rob) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxProfit tests:";
    TIMER_START(maxProfit);
    maxProfit_scaffold("[1, 2, 3, 0, 2]", 3);
    TIMER_STOP(maxProfit);
    util::Log(logESSENTIAL) << "maxProfit using " << TIMER_MSEC(maxProfit) << " milliseconds";

    util::Log(logESSENTIAL) << "Running deleteAndEarn tests:";
    TIMER_START(deleteAndEarn);
    deleteAndEarn_scaffold("[3,4,2]", 6);
    deleteAndEarn_scaffold("[2,2,3,3,3,4]", 9);
    TIMER_STOP(deleteAndEarn);
    util::Log(logESSENTIAL) << "deleteAndEarn using " << TIMER_MSEC(deleteAndEarn) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minSwap tests:";
    TIMER_START(minSwap);
    minSwap_scaffold("[1,3,5,4]", "[1,2,3,7]", 1);
    minSwap_scaffold("[0,3,5,8,9]", "[2,1,4,6,9]", 1);
    TIMER_STOP(minSwap);
    util::Log(logESSENTIAL) << "minSwap using " << TIMER_MSEC(minSwap) << " milliseconds";
}
