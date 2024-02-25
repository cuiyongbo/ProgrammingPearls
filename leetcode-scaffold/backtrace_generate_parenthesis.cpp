#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 20, 22, 301, 678 */

class Solution {
public:
    bool isValidParenthesisString_20(string s);
    bool isValidParenthesisString_678(string s);
    vector<string> generateParenthesis(int n);
    vector<string> removeInvalidParentheses(const string& s);
};


/*
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    An input string is valid if:
        Open brackets must be closed by the same type of brackets.
        Open brackets must be closed in the correct order.
    Note that an empty string is also considered valid.
*/
bool Solution::isValidParenthesisString_20(string s) {
    map<char, char> m;
    m[')'] = '(';
    m[']'] = '[';
    m['}'] = '{';
    stack<char> st;
    for (auto c: s) {
        if (c == '(' || c == '{' || c == '[') {
            st.push(c);
        } else if (c == ')' || c == '}' || c == ']') {
            if (st.empty() || st.top() != m[c]) {
                return false;
            }
            st.pop();
        }
    }
    return st.empty();
}


/*
    Given a string containing only three types of characters: '(', ')' and '*', write a function to check
    whether this string is valid. We define the validity of a string by these rules:
        Any left parenthesis '(' must have a corresponding right parenthesis ')'.
        Any right parenthesis ')' must have a corresponding left parenthesis '('.
        Left parenthesis '(' must go before the corresponding right parenthesis ')'.
        '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
    An empty string is also valid.
*/
bool Solution::isValidParenthesisString_678(string s) {
    stack<int> left_st; // indices of left parentheses
    stack<int> wildcard_st; // indices of wildcard
    for (int i=0; i<s.size(); ++i) {
        if (s[i] == '(') {
            left_st.push(i);
        } else if (s[i] == '*') {
            wildcard_st.push(i);
        } else if (s[i] == ')') {
            if (!left_st.empty()) {
                left_st.pop();
            } else if (!wildcard_st.empty()) {
                wildcard_st.pop();
            } else {
                return false;
            }
        }
    }
    while (!left_st.empty() && !wildcard_st.empty()) {
        if (left_st.top() < wildcard_st.top()) {
            left_st.pop(); wildcard_st.pop();
        } else {
            break;
        }
    }
    return left_st.empty();
}


/*
    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses. For example, given n=3, a solution set is:
        [
            "((()))",
            "(()())",
            "(())()",
            "()(())",
            "()()()"
        ]
*/
vector<string> Solution::generateParenthesis(int n) {
    string alphabet = "()";
    string candidates;
    vector<string> ans;
    // diff = num_of_left_parentheses - num_of_left_parentheses
    function<void(int, int)> backtrace = [&] (int u, int diff) {
        if (u == 2*n) {
            if (diff == 0) {
                ans.push_back(candidates);
            }
            return;
        }
        for (auto c: alphabet) {
            int p = c=='(' ? 1 : -1;
            candidates.push_back(c);
            if (0 <= diff+p && diff+p <= n) {
                backtrace(u+1, diff+p);
            }
            candidates.pop_back();
        }
    };
    backtrace(0, 0);
    return ans;
}


/*
    Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
    Note: The input string may contain letters other than the parentheses ( and ).
*/
vector<string> Solution::removeInvalidParentheses(const string& s) {
    int max_len = 0;
    string candidate;
    set<string> ans;
    // cur_index, numberOfLeftParenthesis, numberOfRightParenthesis
    function<void(int, int, int)> backtrace = [&] (int u, int l , int r) {
        // r > l means candidate is invalid, no need to go further
        if (r > l || u == s.length()) {
            // find a valid target, then save it when its size is no less than max_len
            if (r == l && candidate.size() >= max_len) {
                if (candidate.size() > max_len) {
                    max_len = candidate.size();
                    ans.clear();
                }
                ans.insert(candidate);
            }
            return;
        }

        // 1. discard s[u]
        if (s[u] == '(' || s[u] == ')') {
            backtrace(u+1, l, r);
        }

        // 2. keep s[u], normal backtrace
        l += (s[u] == '(');
        r += (s[u] == ')');
        candidate.push_back(s[u]);
        backtrace(u+1, l, r);
        candidate.pop_back();
        l -= s[u] == '(';
        r -= s[u] == ')';
    };
    backtrace(0, 0, 0);
    return vector<string>(ans.begin(), ans.end());
}


void isValidParenthesisString_scaffold(string input, bool expectedResult, int func) {
    Solution ss;
    bool actual = false;
    if (func == 20) {
        actual = ss.isValidParenthesisString_20(input);
    } else if (func == 678) {
        actual = ss.isValidParenthesisString_678(input);
    } else {
        util::Log(logERROR) << "parament error, func can ony be a value in [20, 678]";
        return;
    }
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case: " << input << ", expectedResult: " << expectedResult << ", func: " << func << ") passed";
    } else {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ", func: " << func << ") failed, actual: " << actual;
    }
}


void generateParenthesis_scaffold(int input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.generateParenthesis(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);

    // to ease test
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    if (actual == expected) {
        util::Log(logINFO) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild, Actual: ";
        for(const auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}


void removeInvalidParentheses_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> actual = ss.removeInvalidParentheses(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    //util::Log(logINFO) << "actual.size=" << actual.size() << ", expected.size=" << expected.size();
    if (actual == expected) {
        util::Log(logINFO) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isValidParenthesisString tests: ";
    TIMER_START(isValidParenthesisString);
    isValidParenthesisString_scaffold("", true, 20);
    isValidParenthesisString_scaffold("(()", false, 20);
    isValidParenthesisString_scaffold("([])", true, 20);
    isValidParenthesisString_scaffold("(]", false, 20);
    isValidParenthesisString_scaffold("([)]", false, 20);
    isValidParenthesisString_scaffold("[((())), (()()), (())(), ()(()), ()()()]", true, 20);
    isValidParenthesisString_scaffold("", true, 678);
    isValidParenthesisString_scaffold("()", true, 678);
    isValidParenthesisString_scaffold("(*)", true, 678);
    isValidParenthesisString_scaffold("(*))", true, 678);
    isValidParenthesisString_scaffold("*(", false, 678);
    isValidParenthesisString_scaffold("*)", true, 678);
    isValidParenthesisString_scaffold("(*", true, 678);
    isValidParenthesisString_scaffold(")*", false, 678);
    TIMER_STOP(isValidParenthesisString);
    util::Log(logESSENTIAL) << "isValidParenthesisString using " << TIMER_MSEC(isValidParenthesisString) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running generateParenthesis tests: ";
    TIMER_START(generateParenthesis);
    generateParenthesis_scaffold(1, "[()]");
    generateParenthesis_scaffold(2, "[(()), ()()]");
    generateParenthesis_scaffold(3, "[((())), (()()), (())(), ()(()), ()()()]");
    TIMER_STOP(generateParenthesis);
    util::Log(logESSENTIAL) << "generateParenthesis using " << TIMER_MSEC(generateParenthesis) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running removeInvalidParentheses tests: ";
    TIMER_START(removeInvalidParentheses);
    removeInvalidParentheses_scaffold("()())()", "[(())(),()()()]");
    removeInvalidParentheses_scaffold("(a)())()", "[(a())(), (a)()()]");
    //removeInvalidParentheses_scaffold(")(", "[]");
    removeInvalidParentheses_scaffold(")()", "[()]");
    TIMER_STOP(removeInvalidParentheses);
    util::Log(logESSENTIAL) << "removeInvalidParentheses using " << TIMER_MSEC(removeInvalidParentheses) << " milliseconds"; 

}
