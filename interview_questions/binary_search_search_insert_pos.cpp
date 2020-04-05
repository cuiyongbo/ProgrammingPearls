#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 34, 35 */

class Solution 
{
public:
    int searchInsert(vector<int>& nums, int target);
    vector<int> searchRange(vector<int>& nums, int target);

private:
    int lower_bound(vector<int>& nums, int target);
    int upper_bound(vector<int>& nums, int target);
};

int Solution::lower_bound(vector<int>& nums, int target)
{
    int l = 0;
    int r = nums.size();
    while(l < r)
    {
        int m = l + (r-l)/2;
        if(nums[m] < target)
        {
            l = m+1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}

int Solution::upper_bound(vector<int>& nums, int target)
{
    int l = 0;
    int r = nums.size();
    while(l < r)
    {
        int m = l + (r-l)/2;
        if(nums[m] <= target)
        {
            l = m+1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}

int Solution::searchInsert(vector<int>& nums, int target)
{
    /*
        Given a sorted array and a target value, return the index if the target is found. 
        If not, return the index where it would be if it were inserted in order.
    */
    return lower_bound(nums, target);
}

vector<int> Solution::searchRange(vector<int>& nums, int target)
{
    /*
        Given an array of integers nums sorted in ascending order, 
        find the starting and ending position of a given target value.
        Your algorithm’s runtime complexity must be in the order of O(log n).
        If the target is not found in the array, return [-1, -1].
    */

    int l = lower_bound(nums, target);
    int r = upper_bound(nums, target);
    if(0 <= l && l < nums.size() && nums[l] == target)
    {
        return {l, r-1};
    }
    else
    {
        return {-1, -1};
    }
}

void searchInsert_scaffold(string input, int target, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringToIntegerVector(input);
    int actual = ss.searchInsert(nums, target);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void searchRange_scaffold(string input, int target, string expectedResult)
{
    Solution ss;
    auto nums = stringToIntegerVector(input);
    auto expected = stringToIntegerVector(expectedResult);
    auto actual = ss.searchRange(nums, target);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchInsert tests:";
    TIMER_START(searchInsert);
    searchInsert_scaffold("[1]", 2, 1);
    searchInsert_scaffold("[1]", 0, 0);
    searchInsert_scaffold("[1,3,5,6]", 5, 2);
    searchInsert_scaffold("[1,3,5,5,5,6]", 5, 2);
    searchInsert_scaffold("[1,3,5,6]", 4, 2);
    searchInsert_scaffold("[1,3,5,6]", 7, 4);
    searchInsert_scaffold("[1,3,5,6]", 0, 0);
    TIMER_STOP(searchInsert);
    util::Log(logESSENTIAL) << "searchInsert using " << TIMER_MSEC(searchInsert) << " milliseconds";

    util::Log(logESSENTIAL) << "Running searchRange tests:";
    TIMER_START(searchRange);
    searchRange_scaffold("[1]", 2, "[-1,-1]");
    searchRange_scaffold("[1]", 0, "[-1, -1]");
    searchRange_scaffold("[1,3,5,6]", 4, "[-1, -1]");
    searchRange_scaffold("[1,3,5,6]", 5, "[2,2]");
    searchRange_scaffold("[1,3,5,5,5,6]", 5, "[2,4]");
    searchRange_scaffold("[5,7,7,8,8,10]", 8, "[3,4]");
    searchRange_scaffold("[5,7,7,8,8,10]", 6, "[-1,-1]");
    TIMER_STOP(searchRange);
    util::Log(logESSENTIAL) << "searchRange using " << TIMER_MSEC(searchRange) << " milliseconds";

}