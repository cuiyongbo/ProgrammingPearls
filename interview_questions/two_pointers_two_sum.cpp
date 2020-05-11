#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 167, 15, 16 */

typedef vector<int> Interval;

class Solution 
{
public:
    vector<int> twoSum(vector<int>& numbers, int target);
    vector<vector<int>> threeSum(vector<int>& nums);
    int threeSumClosest(vector<int> &num, int target);
};

vector<int> Solution::twoSum(vector<int>& nums, int target)
{
    /*
        Given an array of integers that is already sorted in ascending order, 
        find two numbers such that they add up to a specific target number.

        The function twoSum should return indices of the two numbers such that they 
        add up to the target, where index1 must be less than index2. Please note that
        your returned answers (both index1 and index2) are not zero-based.

        You may assume that each input would have exactly one solution and you may not use the same element twice.

        Input: numbers={2, 7, 11, 15}, target=9
        Output: index1=1, index2=2
    */

    int l = 0; 
    int r = (int)nums.size()-1;
    while(l < r)
    {
        if(nums[l] + nums[r] == target)
        {
            break;
        }
        else if(nums[l] + nums[r] < target)
        {
            l++;
        }
        else
        {
            r--;
        }
    }
    return {l+1, r+1};
}

vector<vector<int>> Solution::threeSum(vector<int>& nums)
{
    /*
        Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
        Find all unique triplets in the array which gives the sum of zero.

        Note: The solution set must not contain duplicate triplets.

        For example, given array S = [-1, 0, 1, 2, -1, -4],
 
        A solution set is:
        [
          [-1, 0, 1],
          [-1, -1, 2]
        ]
    */

    std::sort(nums.begin(), nums.end());

    vector<vector<int>> ans;
    int n = (int)nums.size();
    for(int i=0; i<n-2; i++)
    {
        if(nums[i] > 0) break; // branch pruning
        if(i>0 && nums[i] == nums[i-1]) continue; // skip repeat number at same level

        int l = i+1;
        int r = n-1;
        while(l < r)
        {
            if(nums[l] + nums[r] + nums[i] == 0)
            {
                ans.push_back({nums[i], nums[l++], nums[r--]});
                while (l < r && nums[l] == nums[l - 1]) ++l;
                while (l < r && nums[r] == nums[r + 1]) --r; 
            }
            else if(nums[l] + nums[r] + nums[i] < 0)
            {
                l++;
            }
            else
            {
                r--;
            }
        }
    }
    return ans;
}

int Solution::threeSumClosest(vector<int>& nums, int target)
{
    /*
        Given an array nums of n integers and an integer target, 
        find three integers in nums such that the sum is closest to target. 
        Return the sum of the three integers. You may assume that each input would have exactly one solution.

        Example:
            Given array nums = [-1, 2, 1, -4], and target = 1.
            The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
    */

    std::sort(nums.begin(), nums.end());

    int ans = 0;
    int lastDiff = INT32_MAX;
    int n = (int)nums.size();
    for(int i=0; i<n-2; i++)
    {
        int l = i+1; 
        int r = n-1;
        while(l < r)
        {
            int sum = nums[l] + nums[r] + nums[i];
            if(sum == target) return target;
            int diff = std::abs(sum-target);
            if(diff < lastDiff)
            {
                lastDiff = diff;
                ans = sum;
            }
            (sum > target) ? r-- : l++;
        }
    }
    return ans;
}

void twoSum_scaffold(string input1, int input2, string expectedResult)
{
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.twoSum(A, input2);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << numberVectorToString<int>(actual);
    }
}

void threeSum_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.threeSum(A);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString<int>(s);
    }
}

void threeSumClosest_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.threeSumClosest(A, input2);
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

    util::Log(logESSENTIAL) << "Running twoSum tests: ";
    TIMER_START(twoSum);
    twoSum_scaffold("[1,2]", 3, "[1,2]");
    twoSum_scaffold("[2, 7, 11, 15]", 9, "[1,2]");
    TIMER_STOP(twoSum);
    util::Log(logESSENTIAL) << "twoSum using " << TIMER_MSEC(twoSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running threeSum tests: ";
    TIMER_START(threeSum);
    threeSum_scaffold("[-1,0,1,2,-1,-4]", "[[-1,-1,2],[-1,0,1]]");
    TIMER_STOP(threeSum);
    util::Log(logESSENTIAL) << "threeSum using " << TIMER_MSEC(threeSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running threeSumClosest tests: ";
    TIMER_START(threeSumClosest);
    threeSumClosest_scaffold("[-1, 2, 1, -4]", 1, 2);
    TIMER_STOP(threeSumClosest);
    util::Log(logESSENTIAL) << "threeSumClosest using " << TIMER_MSEC(threeSumClosest) << " milliseconds";
}
