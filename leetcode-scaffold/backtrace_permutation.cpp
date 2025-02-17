#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 46, 47, 784, 943, 996 */

class Solution {
public:
    vector<vector<int>> permute_46(vector<int>& nums);
    vector<vector<int>> permute_47(vector<int>& nums);
    vector<string> letterCasePermutation(string S);
    int numSquarefulPerms(vector<int>& A);
    string shortestSuperstring(vector<string>& A);
};


/* 
    Given an integer array without of duplicates, return all possible unique permutations.
    For example, give an input: [1,2,3], Output:
        [
            [1,2,3],
            [1,3,2],
            [2,1,3],
            [2,3,1],
            [3,1,2],
            [3,2,1]
        ]
*/
vector<vector<int>> Solution::permute_46(vector<int>& nums) {
    int sz = nums.size();
    vector<bool> used(sz, false);
    vector<int> candidate; candidate.reserve(nums.size());
    vector<vector<int>> ans;
    std::function<void(int)> backtrace = [&] (int pos) {
        if (pos == sz) {
            ans.push_back(candidate);
            return;
        }
        // start iteration from beginning at each depth 
        for (int i=0; i<sz; i++) {
            // prune invalid branches
            if (used[i]) {
                continue;
            }
            used[i] = true;
            candidate.push_back(nums[i]);
            backtrace(pos+1); // go to the next depth
            candidate.pop_back();
            used[i] = false;
        }
    };
    backtrace(0);
    return ans;
}


/* 
    Same as permute_46, except that the input array may contain duplicates. 
    For example, given an input: [1,1,2], output:
        [
            [1,1,2],
            [1,2,1],
            [2,1,1]
        ]
*/
vector<vector<int>> Solution::permute_47(vector<int>& nums) {
    int sz = nums.size();
    vector<bool> used(sz, false);
    vector<int> candidate; candidate.reserve(nums.size());
    vector<vector<int>> ans;
    std::function<void(int)> backtrace = [&] (int pos) {
        if (pos == sz) {
            ans.push_back(candidate);
            return;
        }
        for (int i=0; i<sz; i++) {
            // we find a duplicate in the previous position but it is not used, which means it has been used in former iteration at current depth
            if (i>0 && nums[i]==nums[i-1] && !used[i-1]) {
                continue;
            }
            // prune invalid branches
            if (used[i]) {
                continue;
            }
            used[i] = true;
            candidate.push_back(nums[i]);
            backtrace(pos+1); // go to the next depth
            candidate.pop_back();
            used[i] = false;
        }
    };
    backtrace(0);
    return ans;
}


/* 
    Given a string, consisting only of letters or digits, we can transform every letter individual to be lowercase
    or uppercase to create another string. return all possible strings we could create.
    Examples:
        Input: S = "a1b2"
        Output: ["a1b2", "a1B2", "A1b2", "A1B2"]

        Input: S = "3z4"
        Output: ["3z4", "3Z4"]

        Input: S = "12345"
        Output: ["12345"]
*/
vector<string> Solution::letterCasePermutation(string input) {
    int sz = input.size();
    vector<string> ans;
    int diff = 'a' - 'A';
    function<void(int)> backtrace = [&] (int u) {
        ans.push_back(input);
        for (int i=u; i<sz; ++i) {
            if (isdigit(input[i])) {
                continue;
            }
            input[i] = std::islower(input[i]) ? (input[i]-diff) : (input[i]+diff); // change case
            backtrace(i+1); // extend to later position only
            input[i] = std::islower(input[i]) ? (input[i]-diff) : (input[i]+diff); // restore case
        }
    };
    backtrace(0);
    return ans;
}


/*
    Given an array of strings words, return the smallest string that contains each string in words as a substring. If there are multiple valid strings of the smallest length, return any of them.
    You may assume that no string in words is a substring of another string in words.

    Method 1: 
    enumerate every permutation, check if a certain permutaion satisfies the requirement, time complexity: O(n!), n = A.size()
    NOT WORK, we don't need to concatenate words[i] to format a super string, actually we need to crash two string instead of just concatenating them
    
    Method 2: 
    DP: g[i][j] is the cost of appending word[j] after word[i], or weight of edge[i][j].

    We would like find the shortest path to visit each node from 0 to n – 1 once and 
    only once this is so-called the Travelling salesman’s problem which is NP-Complete.

    We can solve it with DP that uses exponential time.

    dp[s][i] := min distance to visit nodes (represented as a binary state s) once and only once and the path ends with node i.
    e.g. dp[7][1] is the min distance to visit nodes (0, 1, 2) and ends with node 1, the possible paths could be (0, 2, 1), (2, 0, 1).

    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)
*/
string Solution::shortestSuperstring(vector<string>& A) {

{ // naive method
    std::sort(A.begin(), A.end(), [] (const string& l, const string& r) {
                                        return l.size() < r.size();});
    string ans;
    string path;
    int sz = A.size();
    vector<bool> used(sz, false);

    auto is_super_string = [&] (const string& path) {
        for (int i=0; i<sz; ++i) {
            if (!used[i] && path.find(A[i]) == string::npos) {
                return false;
            }
        }
        return true;
    };

    function<bool(int)> backtrace = [&] (int cur) {
        if (is_super_string(path)) {
            ans = path;
            return true;
        }
        for (int i=0; i<sz; ++i) {
            if (used[i]) {
                continue;
            }
            if (!ans.empty() && path.size()+A[i].size() > ans.size()) { // prune useless branches
                continue;
            }
            used[i] = true;
            path.append(A[i]);
            if (backtrace(cur+1)) {
                return true;
            }
            path.resize(path.size() - A[i].size());
            used[i] = false;
        }
        return false;
    };
    backtrace(0);
    return ans;
}



}


/*
    Given an array A of non-negative integers, the array is squareful if for every pair of adjacent elements, their sum is a perfect square.
    Return the number of permutations of A that are squareful. Two permutations A1 and A2 differ if and only if there is some index i such that A1[i] != A2[i].
    For example, Given a input array [1,17,8] return 2 ([1,8,17] and [17,8,1] are the valid permutations).
*/
int Solution::numSquarefulPerms(vector<int>& A) {
    std::sort(A.begin(), A.end());
    int ans = 0;
    int sz = A.size();
    vector<bool> used(sz, false);
    vector<int> buffer; buffer.reserve(sz);
    auto is_squareful = [&] (int sum) {
        int a = std::sqrt(sum);
        return a*a == sum;
    };
    function<void(int)> backtrace = [&] (int u) {
        if (u == sz) {
            ++ans;
            return;
        }
        for (int i=0; i<sz; ++i) {
            if (used[i]) {
                continue;
            }
            // skip duplicate permutations
            if (i>0 && A[i-1] == A[i] && !used[i-1]) {
                continue;
            }
            // check whether the next pair would be squareful or not
            if (!buffer.empty() && !is_squareful(buffer.back()+A[i])) {
                continue;
            }
            used[i] = true;
            buffer.push_back(A[i]);
            backtrace(u+1);
            buffer.pop_back();
            used[i] = false;
        }
    };
    backtrace(0);
    return ans;
}


void permute_scaffold(string input, string expectedResult, bool duplicate) {
    Solution ss;
    vector<vector<int>> actual;
    vector<int> nums = stringTo1DArray<int>(input);
    if (duplicate) {
        actual = ss.permute_47(nums);
    } else {
        actual = ss.permute_46(nums);
    }

    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);

    // to ease test
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input, expectedResult, duplicate);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed, actual: ", input, expectedResult, duplicate);
        for (const auto& s: actual) {
            SPDLOG_ERROR(numberVectorToString(s));
        }
    }
}


void letterCasePermutation_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.letterCasePermutation(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);

    // to ease test
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: ", input, expectedResult);
        for (const auto& s: actual) {
            SPDLOG_ERROR(s);
        }
    }
}


void shortestSuperstring_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> food = stringTo1DArray<string>(input);
    string actual = ss.shortestSuperstring(food);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input, expectedResult, actual);
    }
}


void numSquarefulPerms_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input);
    int actual = ss.numSquarefulPerms(A);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    SPDLOG_WARN("Running permute tests:");
    TIMER_START(permute);
    permute_scaffold("[1,2,3]", "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]", false);
    permute_scaffold("[1,2,3]", "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]", true);
    permute_scaffold("[1,2,2]", "[[1,2,2],[2,1,2],[2,2,1]]", true);
    permute_scaffold("[1,2,2,2]", "[[1,2,2,2],[2,1,2,2],[2,2,1,2],[2,2,2,1]]", true);
    TIMER_STOP(permute);
    SPDLOG_WARN("permute using {} ms", TIMER_MSEC(permute));

    SPDLOG_WARN("Running letterCasePermutation tests:");
    TIMER_START(letterCasePermutation);
    letterCasePermutation_scaffold("1234", "[1234]");
    letterCasePermutation_scaffold("a2b4", "[a2b4,A2b4,A2B4,a2B4]");
    letterCasePermutation_scaffold("3z4", "[3z4,3Z4]");
    letterCasePermutation_scaffold("a2b4c5", "[a2b4c5,A2b4c5,A2B4c5,A2B4C5,A2b4C5,a2B4c5,a2B4C5,a2b4C5]");
    TIMER_STOP(letterCasePermutation);
    SPDLOG_WARN("letterCasePermutation using {} ms", TIMER_MSEC(letterCasePermutation));

    SPDLOG_WARN("Running shortestSuperstring tests:");
    TIMER_START(shortestSuperstring);
    shortestSuperstring_scaffold("[alex, loves, leetcode]", "alexlovesleetcode");
    shortestSuperstring_scaffold("[catg, ctaagt, gcta, ttca, atgcatc]", "gctaagttcatgcatc");
    TIMER_STOP(shortestSuperstring);
    SPDLOG_WARN("shortestSuperstring using {} ms", TIMER_MSEC(shortestSuperstring));

    SPDLOG_WARN("Running numSquarefulPerms tests:");
    TIMER_START(numSquarefulPerms);
    numSquarefulPerms_scaffold("[1,8,8,1]", 3);
    numSquarefulPerms_scaffold("[2,2,2]", 1);
    numSquarefulPerms_scaffold("[1,8,17]", 2);
    TIMER_STOP(numSquarefulPerms);
    SPDLOG_WARN("numSquarefulPerms using {} ms", TIMER_MSEC(numSquarefulPerms));
}


/*
1,8,1,8
1,8,8,1
8,1,8,1
*/