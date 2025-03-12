#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 53, 121, 309, 1013 */

class Solution {
public:
    int maxSubArray(vector<int>& nums);
    int maxProfit_121(vector<int>& prices);
    int maxProfit_309(vector<int>& prices);
    bool canThreePartsEqualSum(vector<int>& arr);
    int threePartsEqualSumCount(vector<int>& arr);
};


int Solution::threePartsEqualSumCount(vector<int>& arr) {
/*
    Given an array of integers, return the number of solutions with which we can partition the array into three non-empty parts with equal sums.
    for example,
        input: [0, 0, 0, 0]
        output: 3
*/
    int total = std::accumulate(arr.begin(), arr.end(), 0);
    int target = total/3;
    if (target*3 != total) {
        return 0;
    }
    int sz = arr.size();
    // dp[i] means the number of subarray[j:sz], i<=j<sz, whose sum is equal to target
    vector<int> dp(sz, 0); 
    int count = 0;
    int cur_sum = 0; // suffix sum
    for (int i=sz-1; i>=0; --i) {
        cur_sum += arr[i];
        if (cur_sum == target) {
            count++;
        }
        dp[i] = count;
    }
    int ans = 0;
    cur_sum = 0; // prefix sum
    for (int i=0; i<sz-2; ++i) {
        cur_sum += arr[i];
        if (cur_sum == target) {
            ans += dp[i+2];
        }
    }
    return ans;
}


bool Solution::canThreePartsEqualSum(vector<int>& arr) {
/*
    Given an array of integers arr, return true if we can partition the array into three non-empty parts with equal sums.
    Formally, we can partition the array if we can find indexes i + 1 < j with (arr[0] + arr[1] + ... + arr[i] == arr[i + 1] + arr[i + 2] + ... + arr[j - 1] == arr[j] + arr[j + 1] + ... + arr[arr.length - 1])
*/
    int total = std::accumulate(arr.begin(), arr.end(), 0);
    int sub = total/3;
    if (sub*3 != total) {
        return false;
    }
    int sz = arr.size();
    int i = 0;
    int left = 0;
    for (; i<sz; ++i) {
        left += arr[i];
        if (left == sub) { // stop iteration when left==sub
            break;
        }
    }
    int j = sz-1;
    int right = 0;
    for (; j>i; j--) {
        right += arr[j];
        if (right == sub) { // stop iteration when right==sub
            break;
        }
    }
    return j-i>1 && left==sub && right==sub;
}



int Solution::maxSubArray(vector<int>& nums) {
/*
    Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
    For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
    the contiguous subarray [4,-1,2,1] has the largest sum = 6.
*/
    // dp[i] means the largest sum of the contiguous subarray ending with nums[i]
    // dp[i] = max(dp[i-1]+nums[i], nums[i])

{ // navie solution
    int ans = nums[0];
    int sz = nums.size();
    vector<int> dp = nums; // initialize trivial cases
    for (int i=1; i<sz; ++i) {
        dp[i] = max(dp[i-1]+nums[i], dp[i]);
        //dp[i] = i==0 ? nums[i] : max(dp[i-1]+nums[i], nums[i]);
        ans = max(ans, dp[i]);
    }
    return ans;
}

{ // solution with optimization of space usage
    int ans = nums[0];
    int a = nums[0];
    int n = nums.size();
    for (int i=1; i<n; ++i) {
        a = max(a+nums[i], nums[i]);
        ans = max(ans, a);
    }
    return ans;
}   

}


int Solution::maxProfit_121(vector<int>& prices) {
/*
    Say you have an array for which the i-th element is the price of a given stock on day i.
    If you were only permitted to complete at most one transaction (i.e., buy one and sell 
    one share of the stock), design an algorithm to find the maximum profit.
*/

// dp[i] means maxProfit_121 when selling stock no later than i-th day
// dp[i] = max(dp[i-1], prices[i]-purchase_price), purchase_price = min(prices[k]), 0<=k<i

if (0) {
    int n = prices.size();
    vector<int> dp(n, 0);
    int buy = prices[0]; // initialization: we buy at the first day
    for (int i=1; i<n; i++) {
        dp[i] = max(dp[i-1], prices[i]-buy); // if we sell at day i, we can earn prices[i]-buy
        buy = min(buy, prices[i]); // choose a lower price to buy
    }
    return dp[n-1];
}

{ // solution with optimization of space usage
    int n = prices.size();
    int ans = 0;
    int purchase_price = prices[0];
    for (int i=1; i<n; ++i) {
        ans = max(ans, prices[i]-purchase_price);
        purchase_price = min(purchase_price, prices[i]);
    }
    return ans;
}

}


int Solution::maxProfit_309(vector<int>& prices) {
/*
    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions 
    as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

        You may not engage in multiple transactions at the same day (i.e., you must sell the stock before you buy again).
        After you sell your stock, you cannot buy stock on next day. (i.e., cooldown 1 day)
*/

{
    if (prices.empty()) {
        return 0;
    }
    int n = prices.size();
    // Initialize the DP arrays
    vector<int> hold(n, 0), sold(n, 0), cooldown(n, 0);
    // hold[i] means maxProfit_309 if you buy the stock on day i
    // sold[i] means maxProfit_309 if you sell the stock on day i
    // cooldown[i] means maxProfit_309 if you are in a cooldown period on day i (you sold the stock the day before or haven't done any transaction)
    // trivial cases:
    hold[0] = -prices[0]; // We've bought on the first day
    sold[0] = 0;          // Cannot sell on the first day without buying
    cooldown[0] = 0;      // No cooldown on the first day
    // state transitions
    for (int i = 1; i < n; ++i) {
        // you can buy the stock on day i if you were in a cooldown period or you were already holding the stock
        hold[i] = max(hold[i-1], cooldown[i-1] - prices[i]);
        // you can sell the stock on day i if you were holding the stock the day before
        sold[i] = hold[i-1] + prices[i];
        // you can be in a cooldown period on day i if you were in a cooldown period or you just sold the stock the day before
        cooldown[i] = max(cooldown[i-1], sold[i-1]);
    }
    // The result is the maximum profit on the last day being in sold or cooldown states
    return max(sold[n-1], cooldown[n-1]);
}

{ // optimize space usage
    // sold[i] means maxProfit_309 when sold 
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

}


void maxSubArray_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.maxSubArray(costs);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void maxProfit_scaffold(string input, int expectedResult, int func_no) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = 0;
    if (func_no == 121) {
        actual = ss.maxProfit_121(prices);
    } else if (func_no == 309) {
        actual = ss.maxProfit_309(prices);
    } else {
        SPDLOG_ERROR("func_no must be one in [121, 309], actual: {}", func_no);
        return;
    }
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}, func_no={}) passed", input, expectedResult, func_no);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, func_no={}) failed, actual: {}", input, expectedResult, func_no, actual);
    }
}


void canThreePartsEqualSum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.canThreePartsEqualSum(prices);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void threePartsEqualSumCount_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.threePartsEqualSumCount(prices);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running maxSubArray tests:");
    TIMER_START(maxSubArray);
    maxSubArray_scaffold("[1]", 1);
    maxSubArray_scaffold("[1,-1,-1]", 1);
    maxSubArray_scaffold("[-2,1,-3,4,-1,2,1,-5,4]", 6);
    maxSubArray_scaffold("[5,4,-1,7,8]", 23);
    TIMER_STOP(maxSubArray);
    SPDLOG_WARN("maxSubArray tests use {} ms", TIMER_MSEC(maxSubArray));

    SPDLOG_WARN("Running maxProfit tests:");
    TIMER_START(maxProfit);
    maxProfit_scaffold("[7, 1, 5, 3, 6, 4]", 5, 121);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0, 121);
    maxProfit_scaffold("[1, 2, 3, 0, 2]", 3, 309);
    maxProfit_scaffold("[1]", 0, 309);
    maxProfit_scaffold("[1]", 0, 121);
    TIMER_STOP(maxProfit);
    SPDLOG_WARN("maxProfit tests use {} ms", TIMER_MSEC(maxProfit));

    SPDLOG_WARN("Running canThreePartsEqualSum tests:");
    TIMER_START(canThreePartsEqualSum);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 1);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    canThreePartsEqualSum_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    canThreePartsEqualSum_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    TIMER_STOP(canThreePartsEqualSum);
    SPDLOG_WARN("canThreePartsEqualSum tests use {} ms", TIMER_MSEC(canThreePartsEqualSum));

    SPDLOG_WARN("Running threePartsEqualSumCount tests:");
    TIMER_START(threePartsEqualSumCount);
    threePartsEqualSumCount_scaffold("[0,0,0,0]", 3);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 2);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    threePartsEqualSumCount_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    threePartsEqualSumCount_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    TIMER_STOP(threePartsEqualSumCount);
    SPDLOG_WARN("threePartsEqualSumCount tests use {} ms", TIMER_MSEC(threePartsEqualSumCount));
}
