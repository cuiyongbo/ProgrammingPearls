#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 46, 47, 784 */

class Solution
{
public:
    vector<vector<int>> permute(vector<int>& nums);
    vector<vector<int>> permute2(vector<int>& nums);
    vector<string> letterCasePermutation(string S);
};

vector<vector<int>> Solution::permute(vector<int>& nums)
{
    /* Given a collection of distinct integers, return all possible permutations */

    // not necessary
    std::sort(nums.begin(), nums.end());

    int count = (int)nums.size();
    vector<int> cur;
    vector<vector<int>> ans;
    vector<bool> used(count, false);
    function<void(int)> dfs = [&](int n)
    {
        if(n == count)
        {
            ans.push_back(cur);
            return;
        }

        for(int i=0; i<count; ++i)
        {
            if(used[i]) continue;
            used[i] = true;
            cur.push_back(nums[i]);
            dfs(n+1);
            cur.pop_back();
            used[i] = false;
        }
    };

    dfs(0);
    return ans;
}

vector<vector<int>> Solution::permute2(vector<int>& nums)
{
    /* Given a collection of integers which may contain duplicates, return all possible permutations. */

    // not necesary
    std::sort(nums.begin(), nums.end());

    vector<int> cur;
    vector<vector<int>> ans;
    int count = (int)nums.size();
    vector<bool> used(count, false);
    function<void(int)> dfs = [&](int n)
    {
        if(n == count)
        {
            ans.push_back(cur);
            return;
        }

        for(int i=0; i<count; ++i)
        {
            if(used[i]) continue;

            // Same number can only be used once at the same depth
            if(i>0 && nums[i]==nums[i-1] && !used[i-1]) continue;

            used[i] = true;
            cur.push_back(nums[i]);
            dfs(n+1);
            cur.pop_back();
            used[i] = false;
        }
    };

    dfs(0);

    return ans;
}

vector<string> Solution::letterCasePermutation(string S)
{
    /* 
        Given a string S, we can  transform every letter individual to be lowercase
        or uppercase to create another string. return all possible strings we could create.
    */

    string cur = S;
    vector<string> ans;
    int len = (int)S.length();
    function<void(int)> dfs = [&](int pos)
    {
        if(pos == len)
        {
            ans.push_back(cur);
            return;
        }

        dfs(pos+1);

        if(!std::isalpha(cur[pos])) return;

        cur[pos] ^= (1<<5); 
        dfs(pos+1);
        cur[pos] ^= (1<<5); 
    };

    dfs(0);
    return ans;
}

void permute_scaffold(string input, string expectedResult, bool duplicate)
{
    Solution ss;
    vector<vector<int>> actual;
    vector<int> nums = stringToIntegerVector(input);
    if(duplicate)
    {
        actual = ss.permute2(nums);
    }
    else
    {
        actual = ss.permute(nums);
    }
    
    vector<vector<int>> expected = stringTo2DArray(expectedResult);
    BOOST_ASSERT(actual.size() == pow(2, input.size()));
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ", duplicate: " << duplicate << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ", duplicate: " << duplicate << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

void letterCasePermutation_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<string> actual = ss.letterCasePermutation(input);
    vector<string> expected = toStringArray(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running permute tests:";
    TIMER_START(permute);
    permute_scaffold("[1,2,3]", "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]", false);
    permute_scaffold("[1,2,3]", "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]", true);
    permute_scaffold("[1,2,2]", "[[1,2,2],[2,1,2],[2,2,1]]", true);
    permute_scaffold("[1,2,2,2]", "[[1,2,2,2],[2,1,2,2],[2,2,1,2],[2,2,2,1]]", true);
    TIMER_STOP(permute);
    util::Log(logESSENTIAL) << "permute using " << TIMER_MSEC(permute) << " milliseconds";

    util::Log(logESSENTIAL) << "Running letterCasePermutation tests:";
    TIMER_START(letterCasePermutation);
    letterCasePermutation_scaffold("1234", "[1234]");
    letterCasePermutation_scaffold("a2b4", "[a2b4, a2B4, A2b4, A2B4]");
    TIMER_STOP(letterCasePermutation);
    util::Log(logESSENTIAL) << "letterCasePermutation using " << TIMER_MSEC(letterCasePermutation) << " milliseconds";

}
