#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 719, 768 */

class Solution
{
public:
    int smallestDistancePair(vector<int>& nums, int k);
    vector<int> kthSmallestPrimeFraction(vector<int>& A, int K);

private:
    int smallestDistancePair_bruteForce(vector<int>& nums, int k);
    int smallestDistancePair_binarySearch(vector<int>& nums, int k);
};

int Solution::smallestDistancePair(vector<int>& nums, int k)
{
    /*
        Given an integer array, return the k-th smallest distance among all the pairs. 
        The distance of a pair (A, B) is defined as the absolute difference between A and B.
    */

    int count = (int)nums.size();
    (void)count; // suppress [-Wunused-variable] warning
    BOOST_ASSERT_MSG(0<k && k<(count-1)*count/2, "k is invalid");

    // O(n^2)
    // return smallestDistancePair_bruteForce(nums, k);

    // o(n (log(n))^2)
    return smallestDistancePair_binarySearch(nums, k);
}

int Solution::smallestDistancePair_bruteForce(vector<int>& nums, int k)
{
    int count = (int)nums.size();
    map<int, int> distanceCountMap;
    for(int i=0; i<count; ++i)
    {
        for(int j=i+1; j<count; ++j)
            distanceCountMap[std::abs(nums[i]-nums[j])]++;
    }

    int ans = 0;
    int curCount = 0;
    for(const auto& it: distanceCountMap)
    {
        curCount += it.second;
        if(curCount >= k) 
        {
            ans = it.first;
            break;
        }
    }
    return ans;
}

int Solution::smallestDistancePair_binarySearch(vector<int>& nums, int k)
{
    std::sort(nums.begin(), nums.end());

    int count = (int)nums.size();
    auto workhorse = [&](int s, int dis)
    {
        // distance pairs beginning with nums[s]
        // return the number of pairs with distance no larger than dis
        int l = s;
        int r = count;
        while(l < r)
        {
            int m = l + (r-l)/2;
            (nums[m] - nums[s] <= dis) ? (l = m+1) : (r = m);
        }
        return (l > s) ? (l-s-1) : 0;
    };

    int l = 0;
    int r = nums[count-1] - nums[0] + 1;
    while(l < r)
    {
        int total = 0;
        int m = l + (r-l)/2;
        for(int i=0; i<count; ++i)
        {
            total += workhorse(i, m);
        }
        (total < k) ? (l = m+1) : ( r = m);
    }
    return l;
}

vector<int> Solution::kthSmallestPrimeFraction(vector<int>& nums, int k)
{
    /*
        A sorted list A contains 1, plus some number of primes.  
        Then, for every p < q in the list, we consider the fraction p/q.

        What is the K-th smallest fraction considered?  Return your answer as an array of ints, 
        where answer[0] = p and answer[1] = q.  
    */

    const int n = (int)nums.size();

    double l = 0;
    double r = 1.0;
    while(l < r)
    {
        double m = (l+r)/2;
        int total = 0;
        int p=0, q = 0;
        double max_f = 0.0;
        for(int i=0, j=1; i<n-1; ++i)
        {
            while(j<n && nums[i] > m*nums[j]) ++j;
            if(n == j) break;
            total += (n-j);
            double f = double(nums[i])/nums[j];
            if(f > max_f)
            {
                p = i;
                q = j;
                max_f = f;
            }
        }

        if(total == k)
        {
            return {nums[p], nums[q]};
        }
        if(total < k)
        {
            l = m;
        }
        else
        {
            r = m;
        }
    }
    return {};
}

void smallestDistancePair_scaffold(string input, int target, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.smallestDistancePair(nums, target);
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

void kthSmallestPrimeFraction_scaffold(string input, int target, string expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.kthSmallestPrimeFraction(nums, target);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << numberVectorToString(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running smallestDistancePair tests:";
    TIMER_START(smallestDistancePair);
    smallestDistancePair_scaffold("[1,3,1]", 1, 0);
    smallestDistancePair_scaffold("[1,1,1]", 2, 0);
    smallestDistancePair_scaffold("[1,6,1]", 3, 5);
    TIMER_STOP(smallestDistancePair);
    util::Log(logESSENTIAL) << "smallestDistancePair using " << TIMER_MSEC(smallestDistancePair) << " milliseconds";

    util::Log(logESSENTIAL) << "Running kthSmallestPrimeFraction tests:";
    TIMER_START(kthSmallestPrimeFraction);
    kthSmallestPrimeFraction_scaffold("[1,7]", 1, "[1,7]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 3, "[2,5]");
    TIMER_STOP(kthSmallestPrimeFraction);
    util::Log(logESSENTIAL) << "kthSmallestPrimeFraction using " << TIMER_MSEC(kthSmallestPrimeFraction) << " milliseconds";
}
