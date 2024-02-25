#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 53, 121, 1013 */

class Solution {
public:
    int maxSubArray(vector<int>& nums);
    int maxProfit(vector<int>& prices);
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
    int cur_sum = 0;
    for (int i=sz-1; i>=0; --i) {
        cur_sum += arr[i];
        if (cur_sum == target) {
            count++;
        }
        dp[i] = count;
    }

    int ans = 0;
    cur_sum = 0;
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
    int left = 0;
    int right = 0;

    int i=0;
    for (; i<sz; ++i) {
        left += arr[i];
        if (left == sub) {
            break;
        }
    }
    int j=sz-1;
    for (; j>i; --j) {
        right += arr[j];
        if (right == sub) {
            break;
        }
    }
    return j-i>1 && left == sub && right == sub;
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
    int ans = INT32_MIN;
    int sz = nums.size();
    vector<int> dp = nums;
    for (int i=0; i<sz; ++i) {
        dp[i] = i==0 ? nums[i] : max(dp[i-1]+nums[i], nums[i]);
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

int Solution::maxProfit(vector<int>& prices) {
/*
    Say you have an array for which the i-th element is the price of a given stock on day i.
    If you were only permitted to complete at most one transaction (i.e., buy one and sell 
    one share of the stock), design an algorithm to find the maximum profit.
*/

// dp[i] means maxProfit when selling stock no later than i-th day
// dp[i] = max(dp[i-1], prices[i]-purchase_price), purchase_price = min(prices[k]), 0<=k<i

if (0) { // naive solution
    int n = prices.size();
    vector<int> dp(n, 0);
    int purchase_price = prices[0];
    for (int i=1; i<n; ++i) {
        dp[i] = max(dp[i-1], prices[i]-purchase_price);
        purchase_price = min(purchase_price, prices[i]);
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

void maxSubArray_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.maxSubArray(costs);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void maxProfit_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.maxProfit(prices);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void canThreePartsEqualSum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.canThreePartsEqualSum(prices);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void threePartsEqualSumCount_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.threePartsEqualSumCount(prices);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running maxSubArray tests:";
    TIMER_START(maxSubArray);
    maxSubArray_scaffold("[1]", 1);
    maxSubArray_scaffold("[1,-1,-1]", 1);
    maxSubArray_scaffold("[-2,1,-3,4,-1,2,1,-5,4]", 6);
    maxSubArray_scaffold("[5,4,-1,7,8]", 23);
    TIMER_STOP(maxSubArray);
    util::Log(logESSENTIAL) << "maxSubArray using " << TIMER_MSEC(maxSubArray) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxProfit tests:";
    TIMER_START(maxProfit);
    maxProfit_scaffold("[7, 1, 5, 3, 6, 4]", 5);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0);
    TIMER_STOP(maxProfit);
    util::Log(logESSENTIAL) << "maxProfit using " << TIMER_MSEC(maxProfit) << " milliseconds";

    util::Log(logESSENTIAL) << "Running canThreePartsEqualSum tests:";
    TIMER_START(canThreePartsEqualSum);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 1);
    canThreePartsEqualSum_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    canThreePartsEqualSum_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    canThreePartsEqualSum_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    TIMER_STOP(canThreePartsEqualSum);
    util::Log(logESSENTIAL) << "canThreePartsEqualSum using " << TIMER_MSEC(canThreePartsEqualSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running threePartsEqualSumCount tests:";
    TIMER_START(threePartsEqualSumCount);
    threePartsEqualSumCount_scaffold("[0,0,0,0]", 3);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,-7,9,1,2,0,1]", 2);
    threePartsEqualSumCount_scaffold("[0,2,1,-6,6,7,9,-1,2,0,1]", 0);
    threePartsEqualSumCount_scaffold("[3,3,6,5,-2,2,5,1,-9,4]", 1);
    threePartsEqualSumCount_scaffold("[18,12,-18,18,-19,-1,10,10]", 1);
    TIMER_STOP(threePartsEqualSumCount);
    util::Log(logESSENTIAL) << "threePartsEqualSumCount using " << TIMER_MSEC(threePartsEqualSumCount) << " milliseconds";

}
