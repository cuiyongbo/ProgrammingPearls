#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 53, 121 */

class Solution 
{
public:
    int maxSubArray(vector<int>& nums);
    int maxProfit(vector<int>& prices);
};

int Solution::maxSubArray(vector<int>& nums)
{
    /*
        Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
        For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
        the contiguous subarray [4,-1,2,1] has the largest sum = 6.
    */

    // dp[i] means largestSum ending with nums[i]
    // dp[i] = max(dp[i-1] + nums[i], nums[i])

    int ans = INT32_MIN;
    int n = (int)nums.size();

/*
    vector<int> dp(n, 0);
    for(int i=1; i<n; i++)
    {
        dp[i] = std::max(dp[i-1]+nums[i], nums[i]);
        ans = std::max(ans, dp[i]);
    }
*/    
    int dp1=0, dp2=0;
    for(int i=1; i<n; i++)
    {
        dp2 = std::max(dp1+nums[i], nums[i]);
        dp1 = dp2;
        ans = std::max(ans, dp2);
    }

    return ans;
}

int Solution::maxProfit(vector<int>& prices)
{
    /*
        Say you have an array for which the ith element is the price of a given stock on day i.
        If you were only permitted to complete at most one transaction (ie, buy one and sell 
        one share of the stock), design an algorithm to find the maximum profit.
    */

    int n = (int)prices.size();

    // dp[i] means maxProfit when performing transaction before i
    // dp[i] = max(dp[i-1],prices[i] - buy), buy = min{prices[k]} where k<i

    int buy = prices[0];
    int dp1=0, dp2=0;
    for(int i=1; i<n; i++)
    {
        dp2 = std::max(dp1, prices[i]-buy);
        buy = std::min(buy, prices[i]);
        dp1=dp2;
    }
    return dp2;
}

void maxSubArray_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> costs = stringTo1DArray<int>(input);
    int actual = ss.maxSubArray(costs);
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

void maxProfit_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> prices = stringTo1DArray<int>(input);
    int actual = ss.maxProfit(prices);
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

    util::Log(logESSENTIAL) << "Running maxSubArray tests:";
    TIMER_START(maxSubArray);
    maxSubArray_scaffold("[-2,1,-3,4,-1,2,1,-5,4]", 6);
    TIMER_STOP(maxSubArray);
    util::Log(logESSENTIAL) << "maxSubArray using " << TIMER_MSEC(maxSubArray) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxProfit tests:";
    TIMER_START(maxProfit);
    maxProfit_scaffold("[7, 1, 5, 3, 6, 4]", 5);
    maxProfit_scaffold("[7, 6, 4, 3, 1]", 0);
    TIMER_STOP(maxProfit);
    util::Log(logESSENTIAL) << "maxProfit using " << TIMER_MSEC(maxProfit) << " milliseconds";
}
