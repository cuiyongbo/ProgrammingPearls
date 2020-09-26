#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 17, 39, 40, 77, 78 */

class Solution {
public:
    vector<string> letterCombinations(string digits);
    vector<vector<int>> combinationSum(vector<int>& candidates, int target);
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
    vector<vector<int>> combinationSum3(int k, int n);
    vector<vector<int>> combine(int n, int k);
    vector<vector<int>> subsets(vector<int>& nums);
    vector<vector<int>> subsetsWithDup(vector<int>& nums);
};

vector<string> Solution::letterCombinations(string digits) {
/*
    Given a string containing digits from 2-9 inclusive, 
    return all possible letter combinations that the number could represent.

    A mapping of digit to letters (just like on the telephone buttons) is given below. 
    Note that 1 does not map to any letters.
*/
    vector<vector<char>> d(10);
    d[0] = {' '};
    d[1] = {'@'};
    d[2] = {'a','b','c'};
    d[3] = {'d','e','f'};
    d[4] = {'g','h','i'};
    d[5] = {'j','k','l'};
    d[6] = {'m','n','o'};
    d[7] = {'p','q','r','s'};
    d[8] = {'t','u','v'};
    d[9] = {'w','x','y','z'};

    vector<string> ans;
    function<void(int, string)> dfs = [&](int pos, string cur) {
        if(pos == (int)digits.size()) {
            ans.push_back(cur);
            return;
        }

        for(const auto& c: d[digits[pos] - '0']) {
            cur.push_back(c);
            dfs(pos+1, cur);
            cur.pop_back();
        }
    };

    string cur;
    cur.reserve(digits.size());
    dfs(0, cur);
    return ans;
}

vector<vector<int>> Solution::combinationSum(vector<int>& candidates, int target) {
    /*
        Given a set of candidate numbers (C) (without duplicates) and a target number (T), 
        find all unique combinations in C where the candidate numbers sums to T.

        The same repeated number may be chosen from C unlimited number of times.

        Note:
            All numbers (including target) will be positive integers.
            The solution set must not contain duplicate combinations.
    */

    vector<vector<int>> ans;
    int count = (int)candidates.size();
    function<void(int, int, vector<int>&)> dfs = [&](int pos, int sum, vector<int>& cur) {
        if (sum >= target) {
            if (sum == target) {
                ans.push_back(cur);
            }
            return;
        }
        
        for (int i=pos; i<count; ++i) {
            cur.push_back(candidates[i]);
            dfs(i, sum+candidates[i], cur);
            cur.pop_back();
        }
    };

    // sort is not necessary here, but a sorted array makes tests esaier
    std::sort(candidates.begin(), candidates.end());
    vector<int> cur;
    dfs(0, 0, cur);
    return ans;
}

vector<vector<int>> Solution::combinationSum2(vector<int>& candidates, int target)
{
    /*
        Given a set of candidate numbers (C) and a target number (T), 
        find all unique combinations in C where the candidate numbers sums to T.

        Each number in C may be only used once in the combination.

        Note:
            All numbers (including target) will be positive integers.
            The solution set must not contain duplicate combinations.
    */

    set<vector<int>> ans;
    int count = (int)candidates.size();
    function<void(int, int, vector<int>&)> dfs = [&](int pos, int sum, vector<int>& cur) {
        if (sum >= target) {
            if(sum == target) {
                ans.emplace(cur);
            }
            return;
        }
        
        for(int i=pos; i<count; ++i) {
            cur.push_back(candidates[i]);
            dfs(i+1, sum+candidates[i], cur);
            cur.pop_back();
        }
    };

    std::sort(candidates.begin(), candidates.end());
    vector<int> cur;
    dfs(0, 0, cur);
    return vector<vector<int>>(ans.begin(), ans.end());
}

vector<vector<int>> Solution::combinationSum3(int k, int n) {
    /*
        Find all possible combinations of k numbers that add up to a number n, 
        given that only numbers from 1 to 9 can be used and each combination 
        should be a unique set of numbers.
    */

    vector<vector<int>> ans;
    function<void(int, int, vector<int>&)> dfs = [&](int pos, int sum, vector<int>& cur) {
        if (k <= (int)cur.size() || sum >= n) {
            if(sum == n && k == (int)cur.size()) {
                ans.push_back(cur);
            }
            return;
        }

        for (int i=pos; i<10; ++i) {
            cur.push_back(i);
            dfs(i+1, sum+i, cur);
            cur.pop_back();
        }
    };

    vector<int> cur;
    dfs(1, 0, cur);
    return ans;
}

vector<vector<int>> Solution::combine(int n, int k) {
    /*
        Given two integers n and k, return all possible combinations of k numbers out of 1 … n.
    */

    vector<vector<int>> ans;
    function<void(int, vector<int>&)> dfs = [&](int pos, vector<int>& cur) {
        if (k == (int)cur.size()) {
            ans.push_back(cur);
            return;
        }

        for (int i=pos; i<=n; ++i) {
            cur.push_back(i);
            dfs(i+1, cur);
            cur.pop_back();
        }
    };

    vector<int> cur;
    dfs(1, cur);
    return ans;
}

vector<vector<int>> Solution::subsets(vector<int>& nums) {
    /* 
        Given a set of n distinct integers, return its power set
        Note: the solution must contain no duplicate.
    */

    // not necessay, but would make tests easier
    std::sort(nums.begin(), nums.end());

    vector<int> cur;
    vector<vector<int>> ans;
    int size = (int)nums.size();
    function<void(int)> dfs = [&](int pos) {
        ans.push_back(cur);
        if (pos == size) {
            return;
        }
        for (int i=pos; i<size; ++i) {
            cur.push_back(nums[i]);
            dfs(i+1);
            cur.pop_back();
        }
    };

    dfs(0);    
    return ans;
}

vector<vector<int>> Solution::subsetsWithDup(vector<int>& nums) {
    /* 
        Given a collection of integers that may contains duplicates, return all possible subsets (power set).
        Note: the solution must contain no duplicate.
        Example:
            Input: [1,2,2]
            Output:
            [
                [2],
                [1],
                [1,2,2],
                [2,2],
                [1,2],
                []
            ]
    */

    std::sort(nums.begin(), nums.end());

    vector<int> cur;
    vector<vector<int>> ans;
    int size = (int)nums.size();
    function<void(int)> dfs = [&](int pos) {
        ans.push_back(cur);
        if (pos == size) {
            return;
        }
        for (int i=pos; i<size; ++i) {
            // Same number can only be used once at each depth
            if (i>pos && nums[i] == nums[i-1]) {
                continue;
            }
            cur.push_back(nums[i]);
            dfs(i+1);
            cur.pop_back();
        }
    };

    dfs(0);    
    return ans;
}

void letterCombinations_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.letterCombinations(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

void combinationSum_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> candidates = stringTo1DArray<int>(input1);
    vector<vector<int>> actual = ss.combinationSum(candidates, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

void combinationSum2_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> candidates = stringTo1DArray<int>(input1);
    vector<vector<int>> actual = ss.combinationSum2(candidates, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

void combinationSum3_scaffold(int input1, int input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> actual = ss.combinationSum3(input1, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

void combine_scaffold(int input1, int input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> actual = ss.combine(input1, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if(actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

void subsets_scaffold(string input, string expectedResult, bool duplicate) {
    Solution ss;
    vector<vector<int>> actual;
    vector<int> nums = stringTo1DArray<int>(input);
    if (duplicate) {
        actual = ss.subsetsWithDup(nums);
    } else {
        actual = ss.subsets(nums);
    }
    
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for(const auto& s: actual) {
            util::Log(logERROR) << numberVectorToString(s);
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running letterCombinations tests:";
    TIMER_START(letterCombinations);
    letterCombinations_scaffold("23", "[ad,ae,af,bd,be,bf,cd,ce,cf]");
    TIMER_STOP(letterCombinations);
    util::Log(logESSENTIAL) << "letterCombinations using " << TIMER_MSEC(letterCombinations) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum tests:";
    TIMER_START(combinationSum);
    combinationSum_scaffold("[2,3,6,7]", 7, "[[2,2,3], [7]]");
    TIMER_STOP(combinationSum);
    util::Log(logESSENTIAL) << "combinationSum using " << TIMER_MSEC(combinationSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum2 tests:";
    TIMER_START(combinationSum2);
    combinationSum2_scaffold("[2,3,6,7]", 7, "[[7]]");
    combinationSum2_scaffold("[2,3,5,6,7]", 7, "[[2,5], [7]]");
    combinationSum2_scaffold("[10, 1, 2, 7, 6, 1, 5]", 8, "[[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]");
    TIMER_STOP(combinationSum2);
    util::Log(logESSENTIAL) << "combinationSum2 using " << TIMER_MSEC(combinationSum2) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum3 tests:";
    TIMER_START(combinationSum3);
    combinationSum3_scaffold(3, 7, "[[1,2,4]]");
    combinationSum3_scaffold(3, 9, "[[1,2,6], [1,3,5], [2,3,4]]");
    TIMER_STOP(combinationSum3);
    util::Log(logESSENTIAL) << "combinationSum3 using " << TIMER_MSEC(combinationSum3) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combine tests:";
    TIMER_START(combine);
    combine_scaffold(4, 2, "[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]]");
    TIMER_STOP(combine);
    util::Log(logESSENTIAL) << "combine using " << TIMER_MSEC(combine) << " milliseconds";

    util::Log(logESSENTIAL) << "Running subsets tests:";
    TIMER_START(subsets);
    subsets_scaffold("[1,2,3]", "[[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]", false);
    subsets_scaffold("[1,2,3,4]", "[[],[1],[1,2],[1,2,3],[1,2,3,4],[1,2,4],[1,3],[1,3,4],[1,4],[2],[2,3],[2,3,4],[2,4],[3],[3,4],[4]]", false);
    subsets_scaffold("[1,2,3]", "[[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]", true);
    subsets_scaffold("[1,2,3,4]", "[[],[1],[1,2],[1,2,3],[1,2,3,4],[1,2,4],[1,3],[1,3,4],[1,4],[2],[2,3],[2,3,4],[2,4],[3],[3,4],[4]]", true);
    subsets_scaffold("[1,1,1]", "[[],[1],[1,1],[1,1,1]]", true);
    subsets_scaffold("[1,2,2]", "[[],[1],[1,2],[1,2,2],[2],[2,2]]", true);
    TIMER_STOP(subsets);
    util::Log(logESSENTIAL) << "subsets using " << TIMER_MSEC(subsets) << " milliseconds";
}
