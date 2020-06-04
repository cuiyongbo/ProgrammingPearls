#include "leetcode.h"

using namespace std;
using namespace osrm;

class Solution
{
public:
    int subarraySum(vector<int>& nums, int k);
    int subarraySum_naive(vector<int>& nums, int k);
};


int Solution::subarraySum(vector<int>& nums, int k)
{
    int ans = 0;
    int sum = 0;
    unordered_map<int, int> sumMap;
    sumMap[0] = 1;
    for(auto n: nums)
    {
        sum += n;
        if(sumMap.find(sum-k) != sumMap.end())
        {
            ans += sumMap[sum-k];
        }

        sumMap[sum]++;
    }
    return ans;
}

int Solution::subarraySum_naive(vector<int>& nums, int k)
{
    int ans = 0;
    int size = nums.size();
    vector<int> prefix(size, 0);
    partial_sum(nums.begin(), nums.end(), prefix.begin());

    // sum[i, j] = prefix[j] - prefix[i-1]
    for(int i=0; i<size; ++i)
    {
        for(int j=i; j<size; ++j)
        {
            int sumIJ = (i == 0) ? prefix[j] : prefix[j]-prefix[i-1];
            if(sumIJ == k) ++ans;
        }
    }
    return ans;
}

void subarraySum_scaffold(string input, int k)
{
    vector<int> vi = stringTo1DArray<int>(input);

    Solution ss;
    int ans1 = ss.subarraySum(vi, k);
    int ans2 = ss.subarraySum_naive(vi, k);
    if(ans1 == ans2)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << k << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << k << ") failed";
        util::Log(logERROR) << "expected: " << ans1 << ", actual: " << ans2;
    }
}


int main()
{
    util::LogPolicy::GetInstance().Unmute();

    subarraySum_scaffold("[1,1,1,1]", 2);
    subarraySum_scaffold("[1,1,1,1]", 4);
}
