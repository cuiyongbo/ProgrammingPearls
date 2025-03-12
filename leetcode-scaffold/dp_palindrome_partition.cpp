#include "leetcode.h"

using namespace std;

/* leetcode: 131, 89 */

class Solution {
public:
    vector<vector<string>> partition(string s);
    vector<int> grayCode(int n);

private:
    bool is_palindrome(const string& input) {
        int sz = input.size();
        for (int i=0; i<sz/2; i++) {
            if (input[i] != input[sz-1-i]) {
                return false;
            }
        }        
        return true;
    }
};

vector<vector<string>> Solution::partition(string s) {
/*
    Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
    For example, given an input "aab", one possible output would be [["aa","b"],["a","a","b"]]
*/

if (0) { // naive backtrace version
    int sz = s.size();
    vector<vector<string>> ans;
    vector<string> candidate;
    function<void(int)> dfs = [&] (int u) {
        if (u == sz) {
            ans.push_back(candidate);
            return;
        }
        for (int i=u; i<sz; i++) {
            auto v = s.substr(u, i-u+1);
            if (!is_palindrome(v)) {
                continue;
            }
            candidate.push_back(v);
            dfs(i+1);
            candidate.pop_back();
        }
    };
    dfs(0);
    return ans;
}

{ // backtrace with memoization
    typedef vector<vector<string>> result_type;
    typedef pair<bool,result_type> element_type;
    map<string, element_type> sub_solution_mp;
    sub_solution_mp[""] = make_pair(true, result_type());
    function<element_type(string)> worker = [&] (string input) {
        if (sub_solution_mp.count(input) == 1) { // memoization
            return sub_solution_mp[input];
        }

        result_type ans;
        bool valid = false;
        for (int i=0; i<input.size(); ++i) {
            auto left = input.substr(0, i+1);
            if (!is_palindrome(left)) {
                continue;
            }
            auto right = input.substr(i+1);
            auto rr = worker(right);
            if (rr.first) {
                valid = true;
                if (rr.second.empty()) {
                    vector<string> candidate;
                    candidate.insert(candidate.end(), left);
                    ans.push_back(candidate);
                }
                for (auto& p: rr.second) {
                    vector<string> candidate;
                    candidate.insert(candidate.end(), left);
                    candidate.insert(candidate.end(), p.begin(), p.end());
                    ans.push_back(candidate);
                }
            }
        }
        //cout << input << "(" << valid << ", " << ans.size() << ")" << endl;
        sub_solution_mp[input] = make_pair(valid, ans);
        return sub_solution_mp[input];
    };
    return worker(s).second;
}

}

vector<int> Solution::grayCode(int n) {
/*
    The gray code is a binary numeral system where two successive values differ in only one bit. A gray code sequence must begin with 0.
    Given a non-negative integer n representing the total number of bits in the code, (the maximum would be `2^n - 1`) print the sequence of gray code.
    Note: refer to <hacker's delight> ch13 for further info. 
*/

{ // dp solution with optimization of space usage
    vector<int> ans; ans.reserve(1<<n);
    ans.push_back(0); // trivial case
    for (int i=1; i<=n; i++) {
        // ans is already a sequence where two adjacent values differ in only one bit
        int h = 1 << (i-1);
        int sz = ans.size();
        for (int j=sz-1; j>=0; j--) {
            // add one bit to each element in the reverse order of ans
            ans.push_back(h|ans[j]);
        }
    }
    return ans;
}

{ // naive dp solution
    // dp[i] means the graycode with i bits
    // dp[i] = dp[i-1] + {x|(1<<(i-1)) for x in reversed(dp[i-1])}
    // dp[0] = {0}
    vector<vector<int>> dp(n+1);
    dp[0] = {0}; // trivial cases
    for (int i=1; i<=n; ++i) {
        dp[i] = dp[i-1];
        // dp[i-1] is already a subsequence where two successive values differ in only one bit
        for (int j=dp[i-1].size()-1; j >= 0; --j) {
            dp[i].push_back(dp[i-1][j] | (1<<(i-1)));
        }
    }
    return dp[n];
}

}


void partition_scaffold(string input, string expectedResult) {
    Solution ss;
    auto expected = stringTo2DArray<string>(expectedResult);
    auto actual = ss.partition(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, expectedResult);
        for (const auto& s1: actual) {
            for (const auto& s2: s1) {
                std::cout << s2 << ",";
            }
            std::cout << std::endl;
        }
    }
}


void grayCode_scaffold(int input, string expectedResult) {
    Solution ss;
    auto expected = stringTo1DArray<int>(expectedResult);
    auto actual = ss.grayCode(input);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual:", input, numberVectorToString<int>(actual));
    }
}


int main() {
    SPDLOG_WARN("Running partition tests:");
    TIMER_START(partition);
    partition_scaffold("aab", "[[a,a,b],[aa,b]]");
    partition_scaffold("aba", "[[a,b,a],[aba]]");
    TIMER_STOP(partition);
    SPDLOG_WARN("partition tests use {} ms", TIMER_MSEC(partition));

    SPDLOG_WARN("Running grayCode tests:");
    TIMER_START(grayCode);
    grayCode_scaffold(0, "[0]");
    grayCode_scaffold(1, "[0,1]");
    grayCode_scaffold(2, "[0,1,3,2]");
    grayCode_scaffold(3, "[0,1,3,2,6,7,5,4]");
    grayCode_scaffold(4, "[0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8]");
    TIMER_STOP(grayCode);
    SPDLOG_WARN("grayCode tests use {} ms", TIMER_MSEC(grayCode));
}
