#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 46, 47, 784, 943 */

class Solution
{
public:
    vector<vector<int>> permute(vector<int>& nums);
    vector<vector<int>> permute2(vector<int>& nums);
    vector<string> letterCasePermutation(string S);
    string shortestSuperstring(vector<string>& A);
    int numSquarefulPerms(vector<int>& A);
};

vector<vector<int>> Solution::permute(vector<int>& nums)
{
    /* Given a collection of distinct integers, return all possible permutations */

    // not necessary
    std::sort(nums.begin(), nums.end());

    size_t count = nums.size();
    vector<int> cur;
    vector<vector<int>> ans;
    vector<bool> used(count, false);
    function<void()> dfs = [&]()
    {
        if(cur.size() == count)
        {
            ans.push_back(cur);
            return;
        }

        for(size_t i=0; i<count; ++i)
        {
            if(used[i]) continue;

            used[i] = true;
            cur.push_back(nums[i]);

            dfs();
            
            cur.pop_back();
            used[i] = false;
        }
    };

    dfs();

    return ans;
}

vector<vector<int>> Solution::permute2(vector<int>& nums)
{
    /* Given a collection of integers which may contain duplicates, return all possible permutations. */

    // not necesary
    std::sort(nums.begin(), nums.end());

    vector<int> cur;
    vector<vector<int>> ans;
    size_t count = nums.size();
    vector<bool> used(count, false);
    function<void()> dfs = [&]()
    {
        if(cur.size() == count)
        {
            ans.push_back(cur);
            return;
        }

        for(size_t i=0; i<count; ++i)
        {
            if(used[i]) continue;

            // Same number can only be used once at the same depth
            if(i>0 && nums[i]==nums[i-1] && !used[i-1]) continue;

            used[i] = true;
            cur.push_back(nums[i]);
            dfs();
            cur.pop_back();
            used[i] = false;
        }
    };

    dfs();

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

string Solution::shortestSuperstring(vector<string>& A)
{
    /*
        Given an array A of strings, find any smallest string that contains each string in A as a substring.
        We may assume that no string in A is substring of another string in A.
    */

    /*
        Method 1: Enumeration method  time complexity: O(n!), n = A.size()
        
        Method 2: 
        DP: g[i][j] is the cost of appending word[j] after word[i], or weight of edge[i][j].

        We would like find the shortest path to visit each node from 0 to n – 1 once and 
        only once this is called the Travelling sells man’s problem which is NP-Complete.

        We can solve it with DP that uses exponential time.

        dp[s][i] := min distance to visit nodes (represented as a binary state s) once and only once and the path ends with node i.

        e.g. dp[7][1] is the min distance to visit nodes (0, 1, 2) and ends with node 1, the possible paths could be (0, 2, 1), (2, 0, 1).

        Time complexity: O(n^2 * 2^n)
        Space complexity: O(n * 2^n)
    */



    return "";
}

int Solution::numSquarefulPerms(vector<int>& A)
{
    /*
        Given an array A of non-negative integers, the array is squareful 
        if for every pair of adjacent elements, their sum is a perfect square.

        Return the number of permutations of A that are squareful.  
        Two permutations A1 and A2 differ if and only if there is some index 
        i such that A1[i] != A2[i].

        Try all permutaions
    */

    size_t n = A.size();
    vector<bool> used(n, false);
    vector<int> scaffold;
    scaffold.reserve(n);

    auto isSquareful = [](int x, int y)
    {
        int s = std::sqrt(x+y);
        return s*s == x+y;
    };

    int ans = 0;
    function<void()> backtrace = [&]()
    {
        if(scaffold.size() == n)
        {
            ++ans;
            return;
        }

        for(size_t i=0; i<n; ++i)
        {
            if(used[i]) continue;

            // Same number can only be used once at the same depth
            if(i>0 && A[i]==A[i-1] && !used[i-1]) continue;

            if(scaffold.empty() || isSquareful(scaffold.back(), A[i]))
            {
                used[i] = true;
                scaffold.push_back(A[i]);
                backtrace();
                scaffold.pop_back();
                used[i] = false;
            }
        }
    };

    backtrace();

    return ans;    
}


void permute_scaffold(string input, string expectedResult, bool duplicate)
{
    Solution ss;
    vector<vector<int>> actual;
    vector<int> nums = stringTo1DArray<int>(input);
    if(duplicate)
    {
        actual = ss.permute2(nums);
    }
    else
    {
        actual = ss.permute(nums);
    }
    
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
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
    vector<string> expected = stringTo1DArray<string>(expectedResult);
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

void shortestSuperstring_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<string> food = stringTo1DArray<string>(input);
    string actual = ss.shortestSuperstring(food);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << actual;
    }
}

void snumSquarefulPerms_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input);
    int actual = ss.numSquarefulPerms(A);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << actual;
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

    util::Log(logESSENTIAL) << "Running shortestSuperstring tests:";
    TIMER_START(shortestSuperstring);
    shortestSuperstring_scaffold("[alex, loves, leetcode]", "alexlovesleetcode");
    shortestSuperstring_scaffold("[catg, ctaagt, gcta, ttca, atgcatc]", "gctaagttcatgcatc");
    TIMER_STOP(shortestSuperstring);
    util::Log(logESSENTIAL) << "shortestSuperstring using " << TIMER_MSEC(shortestSuperstring) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numSquarefulPerms tests:";
    TIMER_START(numSquarefulPerms);
    snumSquarefulPerms_scaffold("[2,2,2]", 1);
    snumSquarefulPerms_scaffold("[1,8,17]", 2);
    TIMER_STOP(numSquarefulPerms);
    util::Log(logESSENTIAL) << "numSquarefulPerms using " << TIMER_MSEC(numSquarefulPerms) << " milliseconds";

}