#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 20, 22, 301, 678 */

class Solution
{
public:
    bool isValidParenthesisString(string s);
    bool checkValidString(string s);
    vector<string> generateParenthesis(int n);
    vector<string> removeInvalidParentheses(const string& s);

private:
    vector<string> removeInvalidParentheses_01(const string& s);
    vector<string> removeInvalidParentheses_02(const string& s);
    bool checkValidString_counting(string s);
    bool checkValidString_dp(string s);
};

bool Solution::isValidParenthesisString(string s)
{
    /*
        Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
        An input string is valid if:
            Open brackets must be closed by the same type of brackets.
            Open brackets must be closed in the correct order.
        Note that an empty string is also considered valid.
    */

    map<char, char> m;
    m[')'] = '(';
    m[']'] = '[';
    m['}'] = '{';

    const string leftParenthesis = "([{";
    const string rightParenthesis = "}])";
    stack<char> st;
    for(const auto c: s)
    {
        if(leftParenthesis.find(c) != string::npos)
        {
            st.push(c);
        }
        else if(rightParenthesis.find(c) != string::npos)
        {
            if(st.empty() || st.top() != m[c]) return false;
            st.pop();
        }
    }
    return st.empty();
}

bool Solution::checkValidString(string s)
{
    /*
        Given a string containing only three types of characters: '(', ')' and '*', write a function to check
        whether this string is valid. We define the validity of a string by these rules:
            Any left parenthesis '(' must have a corresponding right parenthesis ')'.
            Any right parenthesis ')' must have a corresponding left parenthesis '('.
            Left parenthesis '(' must go before the corresponding right parenthesis ')'.
            '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
        An empty string is also valid.
    */

    // return checkValidString_counting(s);
    return checkValidString_dp(s);
}

bool Solution::checkValidString_dp(string s)
{
    int len = (int)s.length();
    vector<vector<int>> dp (len, vector<int>(len, -1));
    
    // dp[i][j] means if s[i, j] is a valid string
    // s[i, j] is valid if
    //   s[i], [j] is valid pair, and s[i+1, j-1] is valid or
    //   some p in [i, j], s[i, p], s[p+1, j] are valid

    function<bool(int, int)> isValid = [&] (int i, int j)
    {
        // trivial case: empty string
        if(i > j) return true;

        // memorization
        if(dp[i][j] >= 0) return dp[i][j] == 1;

        // trivial case: single character
        if(i == j)
        {
            dp[i][j] = s[i] == '*';
            return dp[i][j] == 1;
        }

        if((s[i]=='(' || s[i] == '*') &&
           (s[j]==')' || s[j] == '*') &&
            isValid(i+1, j-1))
        {
            dp[i][j] = 1;
            return true;
        }

        for(int p=i; p<j; p++)
        {
            if(isValid(i, p) && isValid(p+1, j))
            {
                dp[i][j] = 1;
                return true;
            }
        }

        dp[i][j] = 0;
        return false;
    };

    return isValid(0, len-1);
}

bool Solution::checkValidString_counting(string s)
{
    int min_op = 0, max_op = 0;
    for(const auto& c: s)
    {
        min_op = c=='(' ? min_op+1 : min_op-1;
        max_op = c==')' ? max_op-1 : max_op+1;
        if(max_op < 0) return false;
        min_op = std::max(0, min_op); 
    }
    return min_op == 0;
}

vector<string> Solution::generateParenthesis(int n)
{
    /*
        Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    */

    // method 1: try all permutations, return only the valid unique ones

    vector<string> ans;
    function<void(int,int,string&)> dfs = [&](int l, int r, string& s)
    {
        if(l + r == 0)
        {
            ans.push_back(s);
            return;
        }

        // skip invalid parenthesis
        if(r < l) return;

        if(l > 0)
        {
            s.push_back('(');
            dfs(l-1, r, s);
            s.pop_back();
        }

        if(r > 0)
        {
            s.push_back(')');
            dfs(l, r-1, s);
            s.pop_back();
        }
    };

    string s;
    dfs(n, n, s);

    // not necessary, but grease test
    std::sort(ans.begin(), ans.end());

    return ans;
}

vector<string> Solution::removeInvalidParentheses(const string& s)
{
    /*
        Remove the minimum number of invalid parentheses in order to make the input string valid.
        Return all possible results.
        Note: The input string may contain letters other than the parentheses ( and ).
    */

    return removeInvalidParentheses_02(s);
}

vector<string> Solution::removeInvalidParentheses_02(const string& s)
{
    int l = 0;
    int r = 0;
    size_t maxlen = 0;
    for(const auto& c: s)
    {
        if(c == '(')
            l++;
        else if(c == ')')
            r = (r+1 > l) ? r : r+1;
        else
            maxlen++;
    }

    l = r;
    maxlen += 2*r;
    util::Log(logESSENTIAL) << "maxlen: " << maxlen;
    if (maxlen == 0) return vector<string>();

    int len = (int)s.length();
    vector<string> candidates;
    function<void(int, int, int, string&)> backtrace = [&](int l, int r, int start, string& cur)
    {
        // l, r means the number of parentheses left
        if(l==0 && r==0 && maxlen == cur.size())
        {
            candidates.push_back(cur);
            return;
        }

        // skip invalid parenthesis
        if(l > r) return;

        for (int i = start; i < len; i++)
        {
            if(s[i] == ')')
                r = r - 1;
            else if(s[i] == '(')
                l = l - 1;

            cur.push_back(s[i]);
            backtrace(l, r, i+1, cur);
            cur.pop_back();

            if(s[i] == ')')
                r = r + 1;
            else if(s[i] == '(')
                l = l + 1;
        }
    };
    
    string cur;
    backtrace(l, r, 0, cur);
    std::sort(candidates.begin(), candidates.end());
    candidates.resize(std::unique(candidates.begin(), candidates.end()) - candidates.begin());
    return candidates;
}

vector<string> Solution::removeInvalidParentheses_01(const string& s)
{
    size_t maxlen = 0;
    int len = (int)s.length();

    vector<string> candidates;
    function<void(int, int, int, string&)> backtrace = [&](int l, int r, int start, string& cur)
    {
        if(r==l && start == len && cur.length() >= maxlen)
        {
            maxlen = cur.length();
            candidates.push_back(cur);
            return;
        }

        if(r > l) return;

        for (int i = start; i < len; i++)
        {
            if(s[i] == ')')
                r = r + 1;
            else if(s[i] == '(')
                l = l + 1;

            cur.push_back(s[i]);
            backtrace(l, r, i+1, cur);
            cur.pop_back();

            if(s[i] == ')')
                r = r - 1;
            else if(s[i] == '(')
                l = l - 1;
        }
    };

    string cur;
    backtrace(0, 0, 0, cur);
    // util::Log(logESSENTIAL) << "maxlen: " << maxlen;

    vector<string> ans;
    for (const auto& c: candidates)
    {
        if(c.length() == maxlen)
        {
            ans.push_back(c);
        }
    }
    
    std::sort(ans.begin(), ans.end());
    ans.resize(std::unique(ans.begin(), ans.end()) - ans.begin());
    return ans;
}

void isValidParenthesisString_scaffold(string input, bool expectedResult)
{
    Solution ss;
    bool actual = ss.isValidParenthesisString(input);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void checkValidString_scaffold(string input, bool expectedResult)
{
    Solution ss;
    bool actual = ss.checkValidString(input);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void generateParenthesis_scaffold(int input, string expectedResult)
{
    Solution ss;
    vector<string> actual = ss.generateParenthesis(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    if (actual == expected)
    {
        util::Log(logESSENTIAL) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

void removeInvalidParentheses_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<string> actual = ss.removeInvalidParentheses(input);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    if (actual == expected)
    {
        util::Log(logESSENTIAL) << "Case: " << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case: " << input << ", expectedResult: " << expectedResult << ") faild";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isValidParenthesisString tests: ";
    TIMER_START(isValidParenthesisString);
    isValidParenthesisString_scaffold("", true);
    isValidParenthesisString_scaffold("([])", true);
    isValidParenthesisString_scaffold("(]", false);
    isValidParenthesisString_scaffold("[((())), (()()), (())(), ()(()), ()()()]", true);
    TIMER_STOP(isValidParenthesisString);
    util::Log(logESSENTIAL) << "isValidParenthesisString using " << TIMER_MSEC(isValidParenthesisString) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running checkValidString tests: ";
    TIMER_START(checkValidString);
    checkValidString_scaffold("", true);
    checkValidString_scaffold("()", true);
    checkValidString_scaffold("(*)", true);
    checkValidString_scaffold("(*))", true);
    checkValidString_scaffold("*(", false);
    checkValidString_scaffold("*)", true);
    checkValidString_scaffold("(*", true);
    checkValidString_scaffold(")*", false);
    TIMER_STOP(checkValidString);
    util::Log(logESSENTIAL) << "checkValidString using " << TIMER_MSEC(checkValidString) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running generateParenthesis tests: ";
    TIMER_START(generateParenthesis);
    generateParenthesis_scaffold(3, "[((())), (()()), (())(), ()(()), ()()()]");
    TIMER_STOP(generateParenthesis);
    util::Log(logESSENTIAL) << "generateParenthesis using " << TIMER_MSEC(generateParenthesis) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running removeInvalidParentheses tests: ";
    TIMER_START(removeInvalidParentheses);
    removeInvalidParentheses_scaffold("()())()", "[(())(),()()()]");
    removeInvalidParentheses_scaffold("(a)())()", "[(a())(), (a)()()]");
    removeInvalidParentheses_scaffold(")(", "[]");
    TIMER_STOP(removeInvalidParentheses);
    util::Log(logESSENTIAL) << "removeInvalidParentheses using " << TIMER_MSEC(removeInvalidParentheses) << " milliseconds"; 

}
