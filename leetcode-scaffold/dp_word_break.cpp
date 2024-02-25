#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 139, 140 */

class Solution {
public:
    bool wordBreak_139(string s, vector<string>& wordDict);
    vector<string> wordBreak_140(string s, vector<string>& wordDict);
};

bool Solution::wordBreak_139(string input, vector<string>& wordDict) {
/*
    Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
    determine if s can be segmented into a space-separated sequence of one or more dictionary words. 
    You may assume the dictionary does not contain duplicate words.
*/

    // dp[i] means s[:i) is in wordDict
    // ans = OR(wordBreak(s[:i]) && wordBreak(s[i+1:])), 0<i<s.size
{
    map<string, bool> dict_set; dict_set[""] = true;
    for (auto& p: wordDict) {
        dict_set[p] = true;
    }
    function<bool(string&)> dfs = [&] (string& u) {
        if (dict_set.count(u) == 1) { // memoization
            return dict_set[u];
        }
        dict_set[u] = false;
        for (int i=1; i<=u.size(); ++i) {
            string ul = u.substr(0, i);
            string ur = u.substr(i);
            if (dfs(ul) && dfs(ur)) {
                dict_set[u] = true;
                break;
            }
        }
        return dict_set[u];
    };
    return dfs(input);
}

}

vector<string> Solution::wordBreak_140(string input, vector<string>& wordDict) {
/*
    Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
    add spaces in s to construct a sentence where each word is a valid dictionary word. 
    You may assume the dictionary does not contain duplicate words. Return all such possible sentences.
*/

{
    set<string> dict_set(wordDict.begin(), wordDict.end()); dict_set.insert("");
    typedef vector<vector<string>> solution_type;
    typedef pair<bool, solution_type> result_type;
    map<string, result_type> sub_solution;
    sub_solution[""] = make_pair(true, solution_type(1, vector<string>(1, "")));
    function<result_type(string&)> dfs = [&] (string& u) {
        if (sub_solution.count(u) == 1) { // memoization
            return sub_solution[u];
        }
        sub_solution[u] = make_pair(false, solution_type());
        for (int i=1; i<=u.size(); ++i) {
            string ul = u.substr(0, i);
            if (dict_set.count(ul) == 0) {
                continue;
            }
            string ur = u.substr(i);
            auto rr = dfs(ur);
            if (rr.first) {
                //cout << "[" << ul << ", " << ur << "]" << endl;
                sub_solution[u].first = true;
                for (auto& r: rr.second) {
                    vector<string> candidate;
                    candidate.insert(candidate.end(), ul);
                    candidate.insert(candidate.end(), r.begin(), r.end());
                    sub_solution[u].second.push_back(candidate);
                }
            }
        }
        return sub_solution[u];
    };

    auto res = dfs(input);

    vector<string> ans;
    if (res.first) {
        for (auto& words: res.second) {
            string candidate;
            for (auto& w: words) {
                if (w.empty()) {
                    continue;
                }
                candidate.append(w);
                candidate.push_back(' ');
            }
            candidate.pop_back();
            ans.push_back(candidate);
        }
    }
    return ans;
}

}

void wordBreak_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    bool actual = ss.wordBreak_139(input1, dict);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void wordBreakII_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    auto actual = ss.wordBreak_140(input1, dict);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& s: actual)  {
            util::Log(logERROR) << "[" << s << "]";
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running wordBreak_139 tests:";
    TIMER_START(wordBreak_139);
    wordBreak_scaffold("leetcode", "[leet,code]", true);
    wordBreak_scaffold("leetcode", "[leet,code,loser]", true);
    wordBreak_scaffold("googlebingbaidu", "[google,bing,baidu]", true);
    wordBreak_scaffold("googlebingbaidu360", "[google,bing,baidu]", false);
    TIMER_STOP(wordBreak_139);
    util::Log(logESSENTIAL) << "wordBreak_139 using " << TIMER_MSEC(wordBreak_139) << " milliseconds";

    util::Log(logESSENTIAL) << "Running wordBreak_140 tests:";
    TIMER_START(wordBreak_140);
    wordBreakII_scaffold("leetcode", "[leet,code]", "[leet code]");
    wordBreakII_scaffold("catsanddog", "[cat,cats,and,sand,dog]", "[cat sand dog,cats and dog]");
    TIMER_STOP(wordBreak_140);
    util::Log(logESSENTIAL) << "wordBreak_140 using " << TIMER_MSEC(wordBreak_140) << " milliseconds";
}
