#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 17, 39, 40, 216, 77, 78, 90 */

class Solution {
public:
    vector<string> letterCombinations(string digits);
    vector<vector<int>> combinationSum_39(vector<int>& candidates, int target);
    vector<vector<int>> combinationSum_40(vector<int>& candidates, int target);
    vector<vector<int>> combinationSum_216(int k, int n);
    vector<vector<int>> combine(int n, int k);
    vector<vector<int>> subsets_78(vector<int>& nums);
    vector<vector<int>> subsets_90(vector<int>& nums);
};


/*
    Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
    A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
*/
vector<string> Solution::letterCombinations(string digits) {
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
    string candidate;
    vector<string> ans;
    function<void(int)> backtrace = [&] (int cur) {
        if (cur == digits.size()) {
            ans.push_back(candidate);
            return;
        }
        for (auto c: d[digits[cur]-'0']) {
            candidate.push_back(c);
            backtrace(cur+1);
            candidate.pop_back();
        }
    };
    backtrace(0);
    return ans;
}


/*
    Given a set of candidate numbers **without duplicates** and a target number, find all unique combinations in candidates where the candidate numbers sums to target.
    For example, given inputs: candidates = [2,3,6,7], target = 7, A solution would be [[7],[2,2,3]]

    Note:
        The same repeated number may be chosen from candidate unlimited number of times. 
        All numbers (including target) will be positive integers.
        The solution set must not contain duplicate combinations.
*/
vector<vector<int>> Solution::combinationSum_39(vector<int>& candidates, int target) {
    vector<int> buffer;
    vector<vector<int>> ans;
    int sz = candidates.size();
    function<void(int, int)> backtrace = [&] (int cur, int cur_sum) {
        if (cur_sum == target) {
            ans.push_back(buffer);
            return;
        }
        for (int i=cur; i<sz; ++i) {
            if (cur_sum+candidates[i] > target) {
                continue;
            }
            buffer.push_back(candidates[i]);
            backtrace(i, cur_sum+candidates[i]);
            buffer.pop_back();
        }
    };
    backtrace(0, 0);
    return ans;
}


/*
    Given an array of numbers which may **contain duplicates** and a target number, find all unique combinations in the array where the candidate numbers sums to target. 
    Note:
        Each number in the array may be only used once in the combination.
        All numbers (including target) will be positive integers.
        The solution set must not contain duplicate combinations.
*/
vector<vector<int>> Solution::combinationSum_40(vector<int>& candidates, int target) {
    // preprocess
    sort(candidates.begin(), candidates.end());

    vector<int> buffer;
    vector<vector<int>> ans;
    int sz = candidates.size();
    function<void(int, int)> backtrace = [&] (int u, int sum) {
        if (sum == target) {
            ans.push_back(buffer);
            return;
        }
        for (int i=u; i<sz; ++i) {
            if (sum + candidates[i] > target) {
                break;
            }
            // the same number can ONLY be used once at each depth
            if (i>u && candidates[i-1] == candidates[i]) {
                continue;
            }
            buffer.push_back(candidates[i]);
            backtrace(i+1, sum+candidates[i]);
            buffer.pop_back();
        }
    };
    backtrace(0, 0);
    return ans;
}


/*
    Find all possible combinations of k numbers that add up to a number n , given that only numbers from 1 to 9 can be used and each number is used at most once..
    Ensure that numbers within the set are sorted in ascending order.
    for example,
        Input: k = 3, n = 9
        Output: [[1,2,6], [1,3,5], [2,3,4]]
*/
vector<vector<int>> Solution::combinationSum_216(int k, int n) {
    vector<int> buffer;
    vector<vector<int>> ans;
    function<void(int, int)> backtrace = [&] (int u, int sum) {
        if (buffer.size() == k) {
            if (sum == n) {
                ans.push_back(buffer);
            }
            return;
        }
        for (int i=u; i<10; ++i) {
            if (sum+i > n) {
                break;
            }
            buffer.push_back(i);
            backtrace(i+1, sum+i);
            buffer.pop_back();
        }
    };
    backtrace(1, 0);
    return ans;
}


/*
    Given two integers n and k, return all possible combinations of k numbers out of 1 … n.
    For example, If n=4 and k=2, a solution is [[2,4],[3,4],[2,3],[1,2],[1,3],[1,4]]
*/
vector<vector<int>> Solution::combine(int n, int k) {
    vector<int> buffer;
    vector<vector<int>> ans;
    function<void(int)> backtrace = [&] (int cur) {
        if (buffer.size() == k) {
            ans.push_back(buffer);
            return;
        }
        for (int i=cur; i<=n; ++i) {
            buffer.push_back(i);
            backtrace(i+1);
            buffer.pop_back();
        }
    };
    backtrace(1);
    return ans;
}


/* 
    Given an interger array nums **without duplicates**, return all possible subsets.
    for example, given nums=[1,2,3], the possible solution is
        [
            [],
            [1],
            [1,2],
            [1,2,3],
            [1,3],
            [2],
            [2,3],
            [3]
        ]
    Note:
        **Elements in a subset must be in non-descending order.** (sort the array before starting shuffling nums)
        The solution set must not contain duplicate subsets.
*/
vector<vector<int>> Solution::subsets_78(vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    int sz = nums.size();
    vector<int> buffer; buffer.reserve(sz);
    vector<vector<int>> ans;
    function<void(int)> backtrace = [&] (int u) {
        ans.push_back(buffer);
        for (int i=u; i<sz; ++i) {
            buffer.push_back(nums[i]);
            backtrace(i+1);
            buffer.pop_back();
        }
    };
    backtrace(0);
    return ans;
}


/* 
    Same as subsets_78, except that the input array **may contain duplicates**.
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
vector<vector<int>> Solution::subsets_90(vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    int sz = nums.size();
    vector<int> buffer; buffer.reserve(sz);
    vector<vector<int>> ans;
    function<void(int)> backtrace = [&] (int u) {
        ans.push_back(buffer);
        for (int i=u; i<sz; ++i) {
            // the same number can ONLY be used once at each depth
            if (i>u && nums[i-1] == nums[i]) {
                continue;
            }
            buffer.push_back(nums[i]);
            backtrace(i+1);
            buffer.pop_back();
        }
    };
    backtrace(0);
    return ans;
}


void letterCombinations_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.letterCombinations(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for (const auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}


void combinationSum_scaffold(string input1, int input2, string expectedResult, int func) {
    Solution ss;
    vector<int> candidates = stringTo1DArray<int>(input1);
    vector<vector<int>> actual;
    if (func == 39) {
        actual = ss.combinationSum_39(candidates, input2);
    } else if (func == 40) {
        actual = ss.combinationSum_40(candidates, input2);
    } else {
        util::Log(logERROR) << "argument error, func can only be value in [39|40]";
        return;
    }
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for (const auto& s: actual) {
            util::Log(logERROR) << numberVectorToString(s);
        }
    }
}


void combinationSum_216_scaffold(int input1, int input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> actual = ss.combinationSum_216(input1, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for (const auto& s: actual) {
            util::Log(logERROR) << numberVectorToString(s);
        }
    }
}


void combine_scaffold(int input1, int input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> actual = ss.combine(input1, input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for (const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}


void subsets_scaffold(string input, string expectedResult, bool contain_duplicate) {
    Solution ss;
    vector<vector<int>> actual;
    vector<int> nums = stringTo1DArray<int>(input);
    if (contain_duplicate) {
        actual = ss.subsets_90(nums);
    } else {
        actual = ss.subsets_78(nums);
    }
    
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", contain_duplicate: " << contain_duplicate <<  ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logINFO) << "Case(" << input << ", contain_duplicate: " << contain_duplicate <<  ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:";
        for (const auto& s: actual) {
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

    util::Log(logESSENTIAL) << "Running combinationSum_39 tests:";
    TIMER_START(combinationSum_39);
    combinationSum_scaffold("[2,3,6,7]", 7, "[[2,2,3], [7]]", 39);
    combinationSum_scaffold("[2,3,5]", 8, "[[2,2,2,2], [2,3,3], [3,5]]", 39);
    TIMER_STOP(combinationSum_39);
    util::Log(logESSENTIAL) << "combinationSum_39 using " << TIMER_MSEC(combinationSum_39) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum_40 tests:";
    TIMER_START(combinationSum_40);
    combinationSum_scaffold("[2,3,6,7]", 7, "[[7]]", 40);
    combinationSum_scaffold("[2,3,5,6,7]", 7, "[[2,5], [7]]", 40);
    combinationSum_scaffold("[10, 1, 2, 7, 6, 1, 5]", 8, "[[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]", 40);
    combinationSum_scaffold("[2,5,2,1,2]", 5, "[[1,2,2],[5]]", 40);
    TIMER_STOP(combinationSum_40);
    util::Log(logESSENTIAL) << "combinationSum_40 using " << TIMER_MSEC(combinationSum_40) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum_216 tests:";
    TIMER_START(combinationSum_216);
    combinationSum_216_scaffold(3, 7, "[[1,2,4]]");
    combinationSum_216_scaffold(3, 9, "[[1,2,6], [1,3,5], [2,3,4]]");
    combinationSum_216_scaffold(4, 1, "[]");
    TIMER_STOP(combinationSum_216);
    util::Log(logESSENTIAL) << "combinationSum_216 using " << TIMER_MSEC(combinationSum_216) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combine tests:";
    TIMER_START(combine);
    combine_scaffold(4, 2, "[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]]");
    combine_scaffold(3, 2, "[[1,2],[1,3],[2,3]]");
    combine_scaffold(2, 2, "[[1,2]]");
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
