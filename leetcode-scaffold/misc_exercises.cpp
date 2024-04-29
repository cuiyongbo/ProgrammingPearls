#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 150,224,227,772, 3, 223, 836, 189, 56 */
class Solution {
public:
    int evalRPN(vector<string>& tokens);
    int calculate_224(string s);
    int calculate_227(string s);
    int calculate_772(string s);
    int lengthOfLongestSubstring(string s);
    int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2);
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2);
    void rotate(vector<int>& nums, int k);
    vector<vector<int>> merge(vector<vector<int>>& intervals);

private:
    int calculate_227_infix2postfix(string s);
    int calculate_227_inplace(string s);

    // convert infix notation to postfix notation using Shunting-yard
    // refer to for further detail: https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    vector<string> infix_to_postfix_227(string s);
    int evaluate_postfix_notation(const vector<string>& tokens);

private:
    bool is_digit(char c) { 
        return c >= '0' && c <= '9';
    }
    bool is_operator(char c) { 
        return c == '+' || c == '-' || 
                        c == '*' || c == '/';
    }
    int evaluate(int op1, int op2, char op);

private:
    map<string, int> op_priority {
        {"+", 0}, {"-", 0},
        {"*", 1}, {"/", 1},
        {"a", 2}, // unary plus
        {"A", 2}, // unary minus
        {"(", INT32_MAX}, {")", INT32_MAX},
    };
};


vector<vector<int>> Solution::merge(vector<vector<int>>& intervals) {
/*
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
    Input: intervals = [[1,4],[4,5]]
    Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.
*/
    auto cmp = [](const vector<int>& a, const vector<int>& b) {
        if (a[0] < b[0]) {
            return true;
        } else if (a[0] == b[0]) {
            return a[1] < b[1];
        } else {
            return false;
        }
    };
    auto is_overlap = [] (const vector<int>& a, const vector<int>& b) {
        if (b[0] <= a[1] && a[1] <= b[1]) {
            return true;
        } else {
            return false;
        }
    };

    // sort intervals by left boundary in ascending order
    std::sort(intervals.begin(), intervals.end(), cmp);

    vector<vector<int>> ans;
    ans.reserve(intervals.size());
    vector<int> tmp = intervals[0];
    for (int i=1; i<intervals.size(); i++) {
        if (is_overlap(tmp, intervals[i])) {
            tmp[1] = std::max(tmp[1], intervals[i][1]);
        } else {
            ans.push_back(tmp);
            tmp = intervals[i];
        }
    }
    ans.push_back(tmp);
    return ans;
}

void merge_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> intervals = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.merge(intervals);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& row: actual) {
            util::Log(logERROR) << numberVectorToString<int>(row);
        }
    }
}


void Solution::rotate(vector<int>& nums, int k) {
/*
    Given an array, rotate the array to the right by k steps, where k is non-negative.
    Example 1:
        Input: nums = [1,2,3,4,5,6,7], k = 3
        Output: [5,6,7,1,2,3,4]
        Explanation:
        rotate 1 steps to the right: [7,1,2,3,4,5,6]
        rotate 2 steps to the right: [6,7,1,2,3,4,5]
        rotate 3 steps to the right: [5,6,7,1,2,3,4]
    Example 2:
        Input: nums = [-1,-100,3,99], k = 2
        Output: [3,99,-1,-100]
*/

    auto reverse_worker = [&] (int s, int e) {
        for (int i=0; i<(e-s); ++i) {
            if (s+i >= e-i-1) {
                break;
            }
            swap(nums[s+i], nums[e-i-1]);
        }
    };

    int n = nums.size();
    k %= n;
    reverse_worker(0, n-k);
    reverse_worker(n-k, n);
    reverse_worker(0, n);
}

void rotate_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    vector<int> v2 = stringTo1DArray<int>(expectedResult);
    ss.rotate(v1, input2);
    if(v1 == v2) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expected: " << expectedResult << ") failed, actual: " << numberVectorToString(v1);
    }
}

bool Solution::isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
/*
    An axis-aligned rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) is the coordinate of its bottom-left corner, 
    and (x2, y2) is the coordinate of its top-right corner. Its top and bottom edges are parallel to the X-axis, and its left and right edges are parallel to the Y-axis.
    Two rectangles overlap if the area of their intersection is positive. To be clear, two rectangles that only touch at the corner or edges do not overlap.
    Given two axis-aligned rectangles rec1 and rec2, return true if they overlap, otherwise return false.
*/
    if (rec1[0] >= rec2[2] || rec2[0] >= rec1[2] || // kept away from x-axis
        rec1[1] >= rec2[3] || rec2[1] >= rec1[3] ) { // kept away from y-axis
        return false;
    } else {
        return true;
    }
}

int Solution::computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
/*
    Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles.
    The first rectangle is defined by its bottom-left corner (ax1, ay1) and its top-right corner (ax2, ay2).
    The second rectangle is defined by its bottom-left corner (bx1, by1) and its top-right corner (bx2, by2).
    Hint: ans = area1 + area2 - joint_area
*/
    auto area = [] (int ax1, int ay1, int ax2, int ay2) {
        return (ax2-ax1)*(ay2-ay1);
    };
    int ans = area(ax1, ay1, ax2, ay2) + area(bx1, by1, bx2, by2);
    int intersection = 0;
    if (ax1>=bx2 || bx1>=ax2 || ay1>=by2 || by1>=ay2) { // no intersection
        intersection = 0;
    } else {
        intersection = area(max(ax1,bx1), max(ay1, by1), min(ax2,bx2), min(ay2, by2));
    }
    return ans - intersection;
}

int Solution::lengthOfLongestSubstring(string str) {
/*
    Given a string s, find the length of the longest substring(not subsequence) without repeating characters.

    Example 1:
        Input: s = "abcabcbb"
        Output: 3
        Explanation: The answer is "abc", with the length of 3.
*/

{ // simplified version
    int ans = 0;
    map<char, int> m; // char, the latest position of char
    int left = 0; // left boudary of substring without repeating characters
    int sz = str.size();
    for (int i=0; i<sz; ++i) {
        if (m.count(str[i])) { // duplicate found
            left = max(left, m[str[i]] + 1); // update left boundary
        } 
        ans = max(ans, i-left+1);
        m[str[i]] = i;
    }
    return ans;
}

{ // keep this solution as following for ease of understanding 
    int ans = 0;
    map<char, int> m; // char, the latest position of char
    int left = 0; // left boundary of substring without duplicate characters
    int sz = str.size();
    for (int i=0; i<sz; ++i) {
        if (m.count(str[i]) != 0) { // duplicate found
            left = max(left, m[str[i]]+1); // update left boundary
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        } else {
            m[str[i]] = i; // update occurrence of str[i] to the latest position
            ans = max(ans, i-left+1);
        }
    }
    return ans;
}
}

int Solution::calculate_227_inplace(string s) {
    stack<int> operand_st;
    stack<string> operator_st;
    auto local_eval = [&] () {
        int op2 = operand_st.top(); operand_st.pop();
        int op1 = operand_st.top(); operand_st.pop();
        char op = operator_st.top()[0]; operator_st.pop(); 
        int res = evaluate(op1, op2, op);
        operand_st.push(res);
    };

    int sz = s.size();
    int left = -1, right = -1;
    for (int i=0; i<sz; ++i) {
        if (is_digit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            // save last operand
            operand_st.push(std::stoi(s.substr(left, right-left+1)));
            left = right = -1;

            string op = s.substr(i, 1);
            if (operator_st.empty()) {
                operator_st.push(op);
            } else if (op_priority[op] > op_priority[operator_st.top()]) {
                operator_st.push(op);
            } else { 
                // 栈顶运算符的优先级不低于待插入运算符的优先级，
                // 需要循环计算当前栈顶的表达式，并保存运算结果
                // 直至新栈顶运算符优先级比待插入运算符的优先级低
                while (!operator_st.empty() && op_priority[op] <= op_priority[operator_st.top()]) {
                    local_eval();
                }
                operator_st.push(op);
            }
        } else {
            // space, go on
        }
    }
    if (left != -1) {
        // save last operand
        operand_st.push(std::stoi(s.substr(left, right-left+1)));
    }
 
    while (!operator_st.empty()) {
        local_eval();
    }
    return operand_st.top();
}

int Solution::calculate_227_infix2postfix(string s) {
    auto tokens = infix_to_postfix_227(s);
    return evaluate_postfix_notation(tokens);
}

int Solution::evaluate_postfix_notation(const vector<string>& tokens) {
    stack<int> st;
    for (const auto& t: tokens) {
        if (is_operator(t[0])) {
            int op2 = st.top(); st.pop();
            int op1 = st.top(); st.pop();
            st.push(evaluate(op1, op2, t[0]));
        } else if (t == "a") { // unary plus
            // do nothing
        } else if (t == "A") { // unary minus
            int op = st.top(); st.pop();
            st.push(-op);
        } else {
            st.push(std::stoi(t));
        }
    }
    return st.top();
}

vector<string> Solution::infix_to_postfix_227(string s) {
    vector<string> ans;
    stack<string> operator_st;
    int sz = s.size();
    int left = -1, right = -1;
    for (int i=0; i<sz; ++i) {
        if (is_digit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            // save last operand
            ans.push_back(s.substr(left, right-left+1));
            left = right = -1;

            string op = s.substr(i, 1);
            if (operator_st.empty()) {
                operator_st.push(op);
            } else if (op_priority[op] > op_priority[operator_st.top()]) {
                operator_st.push(op);
            } else { 
                while (!operator_st.empty() && op_priority[op] <= op_priority[operator_st.top()]) {
                    ans.push_back(operator_st.top());
                    operator_st.pop();
                }
                operator_st.push(op);
            }
        } else {
            // space, go on
        }
    }
    if (left != -1) { // save last operand
        ans.push_back(s.substr(left, right-left+1));
    }
    while (!operator_st.empty()) { // save last operators
        ans.push_back(operator_st.top());
        operator_st.pop();
    }
    return ans;
}

int Solution::evaluate(int op1, int op2, char op) {
    int res = 0;
    switch(op) {
        case '+':
            res = op1 + op2;
            break;
        case '-':
            res = op1 - op2;
            break;
        case '*':
            res = op1 * op2;
            break;
        case '/':
            res = op1 / op2;
            break;  
        default:
            break;              
    }
    return res;
}

int Solution::calculate_772(string s) {
    /*
        Implement a basic calculator to evaluate a simple expression string.
        The expression string contains only non-negative integers, '+', '-', '*', '/' operators, 
        and open '(' and closing parentheses ')'. The integer division should truncate toward zero.
        You may assume that the given expression is always valid. All intermediate results will be in the range of [-2^31, 2^31 - 1].
        Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().
        Constraints:
            1 <= s <= 10^4
            s consists of digits, '+', '-', '*', '/', '(', and ')'.
            s is a valid expression.
    */
  
    // return calculate_224(s);

    // 1. preprocess input: remove space(s)
    vector<string> tokens;
    string ns = s; s.clear();
    std::copy_if(ns.begin(), ns.end(), std::back_inserter(s), [](char c) {return c != ' ';});
    
    // 2. convert infix notation to postfix notation
    stack<string> st; // operator stack
    int l=-1, r=-1;
    int sz = s.size();
    for (int i=0; i<sz; ++i) {
        if (is_digit(s[i])) {
            if (l == -1) {
                l = r = i;
            } else {
                r = i;
            }
        } else if (is_operator(s[i])) {
            if (l != -1) { // save last operand
                tokens.push_back(s.substr(l, r-l+1));
                l = r = -1;
            }

            string op = s.substr(i, 1);
            if (op == "+") { // unary plus
                if (i==0 || s[i-1] == '(') {
                    op = "a";
                }
            }
            if (op == "-") { // unary minus
                if (i==0 || s[i-1] == '(') {
                    op = "A";
                }
            }

            // pop operator(s) in stack, until 
            // 1. stack is exhausted,  
            // 2. open parenthesis '(' is encountered
            // 3. priority of the top of stack is less than op's to insert
            while (!st.empty() && op_priority[op] <= op_priority[st.top()]) {
                if (st.top() == "(") {
                    break;
                }
                tokens.push_back(st.top());
                st.pop();
            }
            st.push(op);
        } else if (s[i] == '(') {
            st.push("(");
        } else if (s[i] == ')') {
            if (l != -1) { // save last operand
                tokens.push_back(s.substr(l, r-l+1));
                l = r = -1;
            }
            // exhaust the stack until encountering '(' 
            while (!st.empty() && st.top() != "(") {
                tokens.push_back(st.top());
                st.pop();
            }
            st.pop(); // pop corresponding '('
        }
    }
    if (l != -1) {// save last operand
        tokens.push_back(s.substr(l, r-l+1));
    }    
    while (!st.empty()) {
        tokens.push_back(st.top());
        st.pop();
    }

    // 3. evaluate postfix notation
    return evaluate_postfix_notation(tokens);
}

int Solution::calculate_227(string s) {
    /*
        Given a string s which represents an expression, evaluate this expression and return its value. 
        The integer division should truncate toward zero.
        You may assume that the given expression is always valid. All intermediate results will be in the range of [-(2^31), (2^31) - 1].
        Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as ``eval()``.

        Constraints:
            1 <= s.length <= 3 * 10^5
            s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.
            '+' is not used as a unary operation (i.e., "+1" is invalid).
            '-' is not used as a unary operation (i.e., "-1" is invalid).
            s represents a valid expression.
            All the integers in the expression are non-negative integers in the range [0, 2^31 - 1].
            The answer is guaranteed to fit in a 32-bit integer.
        Hint:
            solution 1. evaluate infix notation in-place
            solution 2. convert infix notation to postfix notation, then evaluate the converted notation
    */

    // return calculate_227_inplace(s);
    return calculate_227_infix2postfix(s);
}

int Solution::calculate_224(string s) {
    /*
        Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.
        Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

        Constraints:
            1 <= s.length <= 3 * 10^5
            s consists of digits, '+', '-', '(', ')', and ' '.
            s represents a valid expression.
            '+' is not used as a unary operation (i.e., "+1" and "+(2 + 3)" is invalid).
            '-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
            There will be no two consecutive operators in the input.
            Every number and running calculation will fit in a signed 32-bit integer.
    */

    // preprocess: remove spaces
    string ns = s; s.clear();
    std::copy_if(ns.begin(), ns.end(), std::back_inserter(s), [](char c) {return c != ' ';});

    vector<string> tokens;
    int sz = s.size();
    int left=-1, right=-1;
    stack<string> operator_st;
    for (int i=0; i<sz; ++i) {
        if (is_digit(s[i])) {
            if (left == -1) {
                left = right = i;
            } else {
                right = i;
            }
        } else if (is_operator(s[i])) {
            if (left != -1) { // save last operand
                tokens.push_back(s.substr(left, right-left+1));
                left = right = -1;
            }

            string op = s.substr(i, 1);
            if (op == "+") { // for problem 772
                // prefixed by ( or no letter, such as +1, (+1), +(1+2)
                if (i==0 || s[i-1] == '(') {
                    op = "a"; // unary plus
                }
            }
            if (op == "-") {
                // prefixed by ( or no letter, such as -1, (1), -(1+2)
                if (i==0 || s[i-1] == '(') {
                    op = "A"; // unary minus
                }
            }
            
            if (operator_st.empty()) {
                operator_st.push(op);
            } else if (op_priority[op] > op_priority[operator_st.top()]) {
                operator_st.push(op);
            } else {
                while (!operator_st.empty() && op_priority[op] <= op_priority[operator_st.top()]) {
                    if (operator_st.top() == "(") {
                        break;
                    }
                    tokens.push_back(operator_st.top());
                    operator_st.pop();    
                }
                operator_st.push(op);
            }
        } else if (s[i] == '(') {
            operator_st.push("(");
        } else if (s[i] == ')') {
            if (left != -1) { // save last operand
                tokens.push_back(s.substr(left, right-left+1));
                left = right = -1;
            }
            while (operator_st.top() != "(") {
                tokens.push_back(operator_st.top());
                operator_st.pop();
            }
            operator_st.pop();
        }
    }
    if (left != -1) { // save last operand
        tokens.push_back(s.substr(left, right-left+1));
    }
    while (!operator_st.empty()) {
        tokens.push_back(operator_st.top());
        operator_st.pop();
    }
    return evaluate_postfix_notation(tokens);
}

int Solution::evalRPN(vector<string>& tokens) {
    /*
        Evaluate the value of an arithmetic expression in Reverse Polish Notation.
        Valid operators are +, -, *, and /. Each operand may be an integer or another expression.
        Note that division between two integers should truncate toward zero.
        It is guaranteed that the given RPN expression is always valid. That means the expression would 
        always evaluate to a result, and there will not be any division by zero operation.
        Example 1:
            Input: tokens = ["2","1","+","3","*"]
            Output: 9
            Explanation: ((2 + 1) * 3) = 9
    */

    stack<int> st;
    auto evaluate = [&] (const string& t) {
        int res = 0;
        int t2 = st.top(); st.pop();
        int t1 = st.top(); st.pop();
        switch(t[0]) {
            case '+':
                res = t1 + t2;
                break;
            case '-':
                res = t1 - t2;
                break;
            case '*':
                res = t1 * t2;
                break;
            case '/':
                res = t1 / t2;
                break;  
            default:
                break;              
        }
        st.push(res);
    };
    for (auto t : tokens) {
        if (t == "+" || t == "-" ||
            t == "*" || t == "/") {
            evaluate(t);
        } else {
            st.push(std::stoi(t));
        }
    }
   return st.top();
}

void evalRPN_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<string>(input);
    int actual = ss.evalRPN(nums);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void calculate_scaffold(string input, int expectedResult, int test_func) {
    Solution ss;
    int actual = 0;
    if (test_func == 224) {
        actual = ss.calculate_224(input);
    } else if (test_func == 227) {
        actual = ss.calculate_227(input);
    } else if (test_func == 772) {
        actual = ss.calculate_772(input);
    } else {
        util::Log(logERROR) << "parameter error: test_func can only be [224, 227, 772] " << actual;
        return;
    }
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " 
                            << expectedResult << ") failed, " << "actual: " << actual;
    }
}

void lengthOfLongestSubstring_scaffold(string input, int expectedResult) {
    Solution ss;
    int actual = ss.lengthOfLongestSubstring(input);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void computeArea_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    int actual = ss.computeArea(vi[0], vi[1], vi[2], vi[3], vi[4], vi[5], vi[6], vi[7]);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void isRectangleOverlap_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<int> v1 = stringTo1DArray<int>(input1);
    vector<int> v2 = stringTo1DArray<int>(input2);
    int actual = ss.isRectangleOverlap(v1, v2);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

// 快速幂: https://zhuanlan.zhihu.com/p/95902286
int qpow2(int a, int n) {
    int ans = 1;
    while (n != 0) {
        if (n&1) {
            ans *= a;
        }
        a *= a;
        n >>= 1;
    }
    return ans;
}

// 区间合并: 给出

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running lengthOfLongestSubstring tests:";
    TIMER_START(lengthOfLongestSubstring);
    lengthOfLongestSubstring_scaffold("", 0);
    lengthOfLongestSubstring_scaffold("abba", 2);
    lengthOfLongestSubstring_scaffold("ababab", 2);
    lengthOfLongestSubstring_scaffold("bbbbb", 1);
    lengthOfLongestSubstring_scaffold("abcdef", 6);
    lengthOfLongestSubstring_scaffold("pwwkew", 3);
    lengthOfLongestSubstring_scaffold("dvdf", 3);
    lengthOfLongestSubstring_scaffold("aaabcdddd", 4);
    lengthOfLongestSubstring_scaffold("aaabcddadd", 4);
    TIMER_STOP(lengthOfLongestSubstring);
    util::Log(logESSENTIAL) << "lengthOfLongestSubstring using " << TIMER_MSEC(lengthOfLongestSubstring) << " milliseconds";

    util::Log(logESSENTIAL) << "Running evalRPN tests:";
    TIMER_START(evalRPN);
    evalRPN_scaffold("[2,1,+,3,*]", 9);
    evalRPN_scaffold("[4,13,5,/,+]", 6);
    evalRPN_scaffold("[10,6,9,3,+,-11,*,/,*,17,+,5,+]", 22);
    TIMER_STOP(evalRPN);
    util::Log(logESSENTIAL) << "evalRPN using " << TIMER_MSEC(evalRPN) << " milliseconds";

    util::Log(logESSENTIAL) << "Running calculate_227 tests:";
    TIMER_START(calculate_227);
    calculate_scaffold("1 + 1", 2, 227);
    calculate_scaffold(" 2-1 + 2", 3, 227);
    calculate_scaffold(" 6-4 / 2 ", 4, 227);
    calculate_scaffold("0", 0, 227);
    calculate_scaffold("3+2*2", 7, 227);
    calculate_scaffold(" 3/2 ", 1, 227);
    calculate_scaffold(" 3+5 / 2 ", 5, 227);
    calculate_scaffold("1*2-3/4+5*6-7*8+9/10", -24, 227);
    TIMER_STOP(calculate_227);
    util::Log(logESSENTIAL) << "calculate_227 using " << TIMER_MSEC(calculate_227) << " milliseconds";

    util::Log(logESSENTIAL) << "Running calculate_224 tests:";
    TIMER_START(calculate_224);
    calculate_scaffold("1 + 1", 2, 224);
    calculate_scaffold(" 2-1 + 2", 3, 224);
    calculate_scaffold("(1+(4+5+2)-3)+(6+8)", 23, 224);
    calculate_scaffold(" 6-4 / 2 ", 4, 224);
    calculate_scaffold("2*(5+5*2)/3+(6/2+8)", 21, 224);
    calculate_scaffold("(2+6* 3+5- (3*14/7+2)*5)+3", -12, 224);
    calculate_scaffold("0", 0, 224);
    calculate_scaffold("-1", -1, 224);
    calculate_scaffold("(-1)", -1, 224);
    calculate_scaffold("-(2 + 3)", -5, 224);
    TIMER_STOP(calculate_224);
    util::Log(logESSENTIAL) << "calculate_772 using " << TIMER_MSEC(calculate_224) << " milliseconds";

    util::Log(logESSENTIAL) << "Running calculate_772 tests:";
    TIMER_START(calculate_772);
    calculate_scaffold("1 + 1", 2, 772);
    calculate_scaffold(" 2-1 + 2", 3, 772);
    calculate_scaffold("(1+(4+5+2)-3)+(6+8)", 23, 772);
    calculate_scaffold(" 6-4 / 2 ", 4, 772);
    calculate_scaffold("2*(5+5*2)/3+(6/2+8)", 21, 772);
    calculate_scaffold("(2+6* 3+5- (3*14/7+2)*5)+3", -12, 772);
    calculate_scaffold("0", 0, 772);
    calculate_scaffold("-1", -1, 772);
    calculate_scaffold("-(2 + 3)", -5, 772);
    calculate_scaffold("+1", 1, 772);
    calculate_scaffold("+(2 + 3)", 5, 772);
    TIMER_STOP(calculate_772);
    util::Log(logESSENTIAL) << "calculate_772 using " << TIMER_MSEC(calculate_772) << " milliseconds";

    util::Log(logESSENTIAL) << "Running computeArea tests:";
    TIMER_START(computeArea);
    computeArea_scaffold("[-3,0,3,4,0,-1,9,2]", 45);
    computeArea_scaffold("[-2,-2,2,2,-2,-2,2,2]", 16);
    TIMER_STOP(computeArea);
    util::Log(logESSENTIAL) << "computeArea using " << TIMER_MSEC(computeArea) << " milliseconds";

    util::Log(logESSENTIAL) << "Running isRectangleOverlap tests:";
    TIMER_START(isRectangleOverlap);
    isRectangleOverlap_scaffold("[-3,0,3,4]", "[0,-1,9,2]", 1);
    isRectangleOverlap_scaffold("[-2,-2,2,2]", "[-2,-2,2,2]", 1);
    isRectangleOverlap_scaffold("[-2,-2,2,2]", "[-2,-2,2,2]", 1);
    isRectangleOverlap_scaffold("[0,0,2,2]", "[1,1,3,3]", 1);
    isRectangleOverlap_scaffold("[0,0,1,1]", "[1,0,2,1]", 0);
    isRectangleOverlap_scaffold("[0,0,1,1]", "[2,2,3,3]", 0);
    TIMER_STOP(isRectangleOverlap);
    util::Log(logESSENTIAL) << "isRectangleOverlap using " << TIMER_MSEC(isRectangleOverlap) << " milliseconds";

    util::Log(logESSENTIAL) << "Running rotate tests:";
    TIMER_START(rotate);
    rotate_scaffold("[1,2,3,4,5,6,7]", 3, "[5,6,7,1,2,3,4]");
    rotate_scaffold("[-1,-100,3,99]", 2, "[3,99,-1,-100]");
    TIMER_STOP(rotate);
    util::Log(logESSENTIAL) << "rotate using " << TIMER_MSEC(rotate) << " milliseconds";

    util::Log(logESSENTIAL) << "Running merge tests:";
    TIMER_START(merge);
    merge_scaffold("[[1,3],[2,6],[8,10],[15,18]]", "[[1,6],[8,10],[15,18]]");
    merge_scaffold("[[1,4],[4,5]]", "[[1,5]]");
    TIMER_STOP(merge);
    util::Log(logESSENTIAL) << "merge using " << TIMER_MSEC(merge) << " milliseconds";

}
