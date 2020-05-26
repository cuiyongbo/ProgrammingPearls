#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 139, 140, 818 */

class Solution 
{
public:
    bool wordBreak(string s, vector<string>& wordDict);
    vector<string> wordBreakII(string s, vector<string>& wordDict);
};

bool Solution::wordBreak(string s, vector<string>& wordDict)
{
    /*
        Given a non-empty string s and a dictionary wordDict containing 
        a list of non-empty words, determine if s can be segmented into 
        a space-separated sequence of one or more dictionary words. 
        You may assume the dictionary does not contain duplicate words.
    */

    // dp[i] means s[:i) is in wordDict
    // ans = OR(wordBreak(s[:i]) && wordBreak(s[i+1:]))

    map<string, bool> dict;
    dict[""] = true;
    for(const auto& word: wordDict)
        dict[word] = true;

    function<bool(const string&)> helper = [&](const string& s)
    {
        auto it = dict.find(s);
        if(it != dict.end()) return dict[s];

        for(int i=1; i<s.length(); i++)
        {
            auto left = s.substr(0, i);
            auto right = s.substr(i);
            if(helper(left) && helper(right))
            {
                dict[s] = true;
                return dict[s];
            }
        }
        dict[s] = false;
        return dict[s];
    };

    return helper(s);
}

vector<string> Solution::wordBreakII(string s, vector<string>& wordDict)
{
    /*
        Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
        add spaces in s to construct a sentence where each word is a valid dictionary word. 
        You may assume the dictionary does not contain duplicate words.
        Return all such possible sentences.
    */

    auto make_ans = [](const vector<string> prefixes, const string& word) 
    {
        vector<string> results;
        for(const auto& prefix : prefixes)
            results.push_back(prefix + " " + word);
        return results;
    };

    set<string> dict(wordDict.begin(), wordDict.end());
    map<string, vector<string>> subSolutions;

    function<vector<string>(const string&)> helper = [&](const string& s)
    {
        auto it = subSolutions.find(s);
        if(it != subSolutions.end()) return it->second;

        vector<string> ans;
        if(dict.count(s)) ans.push_back(s);

        for(int i=1; i<s.size(); ++i)
        {
            auto right = s.substr(i);
            if(!dict.count(right)) continue;

            auto left = s.substr(0, i);
            auto left_ans = make_ans(helper(left), right);
            ans.insert(ans.end(), left_ans.begin(), left_ans.end());
        }

        subSolutions[s].swap(ans);
        return subSolutions[s];
    };

    return helper(s);
}

void wordBreak_scaffold(string input1, string input2, bool expectedResult)
{
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    bool actual = ss.wordBreak(input1, dict);
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

void wordBreakII_scaffold(string input1, string input2, string expectedResult)
{
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input2);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    auto actual = ss.wordBreakII(input1, dict);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual)  util::Log(logERROR) << s;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running wordBreak tests:";
    TIMER_START(wordBreak);
    wordBreak_scaffold("leetcode", "[leet,code]", true);
    wordBreak_scaffold("googlebingbaidu", "[google,bing,baidu]", true);
    wordBreak_scaffold("googlebingbaidu360", "[google,bing,baidu]", false);
    TIMER_STOP(wordBreak);
    util::Log(logESSENTIAL) << "wordBreak using " << TIMER_MSEC(wordBreak) << " milliseconds";

    util::Log(logESSENTIAL) << "Running wordBreakII tests:";
    TIMER_START(wordBreakII);
    wordBreakII_scaffold("leetcode", "[leet,code]", "[leet code]");
    wordBreakII_scaffold("catsanddog", "[cat,cats,and,sand,dog]", "[cat sand dog,cats and dog]");
    TIMER_STOP(wordBreakII);
    util::Log(logESSENTIAL) << "wordBreakII using " << TIMER_MSEC(wordBreakII) << " milliseconds";
}
