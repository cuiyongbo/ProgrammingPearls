#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 698, 93, 131, 241, 282, 842 */

namespace tiny_scaffold {
    int add(int a, int b) { return a+b;}
    int sub(int a, int b) { return a-b;};
    int mul(int a, int b) { return a*b;};
}

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k);
    vector<string> restoreIpAddresses(string s);
    vector<vector<string>> partition(string s);
    vector<int> diffWaysToCompute(string input);
    vector<string> addOperators(string num, int target);
    vector<int> splitIntoFibonacci(string s);
};


/*
    Given an array of integers nums and a positive integer k, find whether it’s possible to divide this array into k non-empty subsets whose sums are all equal.
    For example, given inputs: nums = [4, 3, 2, 3, 5, 2, 1], k = 4, It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
*/
bool Solution::canPartitionKSubsets(vector<int>& nums, int k) {
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    int target = sum / k;
    if (target * k != sum) {
        return false;
    }
    int len = nums.size();
    vector<bool> used(len, false);
    function<bool(int, int)> backtrace = [&] (int cur, int partitions) {
        if (partitions==k && std::all_of(used.begin(), used.end(), [](bool v){return v;})) {
            return true;
        }
        for (int i=0; i<len; ++i) {
            if (used[i] || cur+nums[i]>target) { // prune useless branches
                continue;
            }
            used[i] = true;
            cur += nums[i];
            if (cur == target && backtrace(0, partitions+1)) {
                return true;
            } else if (backtrace(cur, partitions)) {
                return true;
            }
            cur -= nums[i];
            used[i] = false;
        }
        return false;
    };
    return backtrace(0, 0);
}


/*
    A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.
    For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.

    Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s.
    You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.
    Example:
        Input: "25525511135"
        Output: ["255.255.11.135", "255.255.111.35"]
*/
vector<string> Solution::restoreIpAddresses(string input) {
    auto is_valid = [] (string tmp) {
        if (tmp.size()>1 && tmp[0] == '0') {
            return false;
        }
        int n = stoi(tmp);
        return 0<=n && n<=255;
    };
    int sz = input.size();
    vector<string> ans;
    vector<string> candidate;
    function<void(int)> backtrace = [&] (int u) {
        if (u == sz || candidate.size()==4) {
            if (u==sz && candidate.size() == 4) {
                ans.push_back(
                        candidate[0] + "." + 
                        candidate[1] + "." + 
                        candidate[2] + "." + 
                        candidate[3]);
            }
            return;
        }
        for (int i=u; i<min(u+4, sz); ++i) {
            string tmp = input.substr(u, i-u+1);
            if (is_valid(tmp)) {
                candidate.push_back(tmp);
                backtrace(i+1);
                candidate.pop_back();
            }
        }
    };
    backtrace(0);
    return ans;
}


/*
    Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
    For example, given an input: "aab", output:[["aa","b"],["a","a","b"]]
*/
vector<vector<string>> Solution::partition(string input) {
    auto is_palindrome = [] (string tmp) {
        int sz = tmp.size();
        for (int i=0; i<sz/2; ++i) {
            if (tmp[i] != tmp[sz-1-i]) {
                return false;
            }
        }
        return true;
    };
    vector<string> candidate;
    vector<vector<string>> ans;
    int sz = input.size();
    function<void(int)> backtrace = [&] (int u) {
        if (u==sz) {
            ans.push_back(candidate);
            return;
        }
        for (int i=u; i<sz; i++) {
            string tmp = input.substr(u, i-u+1);
            if (is_palindrome(tmp)) {
                candidate.push_back(tmp);
                backtrace(i+1);
                candidate.pop_back();
            }
        }
    };
    backtrace(0);
    return ans;
}


/*
    Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators.
    The valid operators are +, - and *.
    Example 1:
        Input: "2-1-1".
        ((2-1)-1) = 0
        (2-(1-1)) = 2
        Output: [0, 2]
    Example 2:
        Input: "2*3-4*5"
        (2*(3-(4*5))) = -34
        ((2*3)-(4*5)) = -14
        ((2*(3-4))*5) = -10
        (2*((3-4)*5)) = -10
        (((2*3)-4)*5) = 10
        Output: [-34, -14, -10, -10, 10]
*/
vector<int> Solution::diffWaysToCompute(string input) {
    map<char, function<int(int, int)>> func_map;
    func_map['+'] = tiny_scaffold::add;
    func_map['-'] = tiny_scaffold::sub;
    func_map['*'] = tiny_scaffold::mul;
    typedef vector<int> result_t;
    map<string, result_t> sub_solutions; // backtrace with memoization
    function<result_t(string)> backtrace = [&] (string input) {
        if (sub_solutions.count(input) != 0) {
            return sub_solutions[input];
        }
        result_t ans;
        bool operator_found = false;
        for (int i=0; i<input.size(); ++i) {
            if (std::isdigit(input[i])) {
                continue;
            }
            operator_found = true;
            // similar to postorder traversal
            string li = input.substr(0, i);
            string ri = input.substr(i+1);
            result_t l = backtrace(li);
            result_t r = backtrace(ri);
            //cout << "input=" << input << ", l.size=" << l.size() << ", r.size=" << r.size() << endl;
            for (auto a: l) {
                for (auto b: r) {
                    ans.push_back(func_map[input[i]](a, b));
                }
            }
        }
        if (!operator_found) { // trivial case, input is an operand
            ans.push_back(std::stod(input));
        }
        sub_solutions[input] = ans;
        return ans;
    };
    return backtrace(input);
}


/*
    Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.
    Examples:
        "123", 6 -> ["1+2+3", "1*2*3"] 
        "232", 8 -> ["2*3+2", "2+3*2"]
        "105", 5 -> ["1*0+5","10-5"]
        "00", 0 -> ["0+0", "0-0", "0*0"]
        "3456237490", 9191 -> []
*/
vector<string> Solution::addOperators(string num, int target) {
    typedef vector<string> result_t;
    result_t operators {"+", "-", "*"};
    map<string, int> priority_map;
    priority_map["+"] = 0;
    priority_map["-"] = 0;
    priority_map["*"] = 1;

    typedef function<int(int, int)> operator_t;
    map<string, operator_t> func_map;
    func_map["+"] = tiny_scaffold::add;
    func_map["-"] = tiny_scaffold::sub;
    func_map["*"] = tiny_scaffold::mul;

    auto is_operator = [] (const string& c) { 
        return c == "-" || c == "+" || c =="*";
    };

    auto inplace_eval = [&] (result_t& infix_exp) {
        stack<int> operand;
        stack<string> op_st;
        for (auto& p: infix_exp) {
            if (is_operator(p)) {
                while (!op_st.empty() && priority_map[op_st.top()] >= priority_map[p]) {
                    auto b = operand.top(); operand.pop();
                    auto a = operand.top(); operand.pop();
                    operand.push(func_map[op_st.top()](a, b));
                    op_st.pop();
                }
                op_st.push(p);
            } else {
                operand.push(stoi(p));
            }
        }
        while(!op_st.empty()) {
            auto b = operand.top(); operand.pop();
            auto a = operand.top(); operand.pop();
            operand.push(func_map[op_st.top()](a, b));
            op_st.pop();
        }
        return operand.top();
    };

    // http://csis.pace.edu/~wolf/CS122/infix-postfix.htm
    auto infix_to_postfix = [&](const result_t& infix_exp) {
        stack<string> op;
        result_t postfix_exp;
        for (const auto& c: infix_exp) {
            if (is_operator(c)) {
                // pop stack when: 
                //      1. the priority of current operator is less than op.top's, such as + is against *
                //      2. the priority of current operator is equal to op.top's, such as + is against -
                if (!op.empty() && priority_map[op.top()] >= priority_map[c]) {
                    postfix_exp.push_back(op.top()); op.pop();
                }
                op.push(c);
            } else {
                postfix_exp.push_back(c);
            }
        }
        while (!op.empty()) {
            postfix_exp.push_back(op.top()); op.pop();
        }
        return postfix_exp;
    };

    auto evaluate = [&](const result_t& exp) {
        stack<int> operands;
        const auto& postfix = infix_to_postfix(exp);
        for (const auto& op: postfix) {
            if (is_operator(op)) {
                auto b = operands.top(); operands.pop();
                auto a = operands.top(); operands.pop();
                auto c = func_map[op](a, b);
                operands.push(c);
            } else {
                operands.push(std::stol(op));
            }
        }
        return operands.top();
    };

    result_t ans;
    result_t exp;
    int len = num.size();
    function<void(int)> backtrace = [&] (int p) {
        if (p == len && inplace_eval(exp) == target) {
        //if (p == len && evaluate(exp) == target) {
            string path;
            for (auto& s: exp) {
                path += s;
            }
            ans.push_back(path);
            return;
        }
        for (int i=p; i<len; ++i) {
            // prune useless branches
            if (i>p && num[p]=='0') { // skip operands with leading zero(s), such as "05"
                continue;
            }
            string cur = num.substr(p, i-p+1);
            if (stol(cur) > INT32_MAX) { // signed integer overflow
                continue;
            }
            exp.push_back(cur);
            if (i+1 == len) { // no need to try more operators since there is no operand left
                backtrace(i+1);
            } else {
                for (auto& op: operators) {
                    exp.push_back(op);
                    backtrace(i+1);
                    exp.pop_back();
                }
            }
            exp.pop_back();
        }
    };

    backtrace(0);
    return ans;
}


/*
    Given a string S of digits, such as S = "123456579", we can split it into a Fibonacci-like sequence [123, 456, 579].
    Formally, a Fibonacci-like sequence is a list F of non-negative integers such that:
        0 <= F[i] <= 2^31 - 1, (that is, each integer fits a 32-bit signed integer type);
        F.length >= 3;
        and F[i] + F[i+1] = F[i+2] for all 0 <= i < F.length - 2.

    Also note that when splitting the string into pieces, each piece must not have extra leading zeroes, except if the piece is the number 0 itself. 
    also the input string contains only digits. Return any Fibonacci-like sequence split from S, or return [] if it cannot be done.

    for example, 
        Input: "11235813"
        Output: [1,1,2,3,5,8,13]
        Input: "112358130"
        Output: []
        Explanation: The task is impossible.
        Input: "0123"
        Output: []
        Explanation: Leading zeroes are not allowed, so "01", "2", "3" is not valid.
*/
vector<int> Solution::splitIntoFibonacci(string input) {
    vector<int> ans;
    int sz = input.size();
    function<bool(int)> backtrace = [&] (int p) {
        if (p == sz) {
            return ans.size() >= 3;
        }
        for (int i=p; i<sz; ++i) {
            // prune useless branches
            if (i>p && input[p]=='0') { // skip element(s) with leading zero(s)
                continue;
            }
            string tmp = input.substr(p, i-p+1);
            long n = std::stol(tmp);
            if (n > INT32_MAX) { // signed integer overflow
                continue;
            }
            int d = ans.size();
            if (d < 2 || (ans[d-2]+ ans[d-1] == n)) {
                ans.push_back(n);
                if (backtrace(i+1)) {
                    return true;
                }
                ans.pop_back();
            } 
        }
        return false;
    };
    backtrace(0);
    return ans;
}

void canPartitionKSubsets_scaffold(string input1, int input2, bool expectedResult) {
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    bool actual = ss.canPartitionKSubsets(g, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void restoreIpAddresses_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.restoreIpAddresses(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}

void partition_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.partition(input);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: ";
        for (auto vs: actual) {
            util::Log(logERROR) << "*********";
            for (auto s: vs) {
                util::Log(logERROR) <<  s;
            }
        }
    }
}

void diffWaysToCompute_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.diffWaysToCompute(input);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, acutal: "<< numberVectorToString<int>(actual);
    }
}

void addOperators_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.addOperators(input1, input2);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}

void splitIntoFibonacci_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<int> actual = ss.splitIntoFibonacci(input);
    if (!actual.empty() == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << !actual.empty();
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running canPartitionKSubsets tests: ";
    TIMER_START(canPartitionKSubsets);
    canPartitionKSubsets_scaffold("[4,3,2,3,5,2,1]", 4, true);
    TIMER_STOP(canPartitionKSubsets);
    util::Log(logESSENTIAL) << "canPartitionKSubsets using " << TIMER_MSEC(canPartitionKSubsets) << " milliseconds";

    util::Log(logESSENTIAL) << "Running restoreIpAddresses tests: ";
    TIMER_START(restoreIpAddresses);
    restoreIpAddresses_scaffold("25525511135", "[255.255.11.135, 255.255.111.35]");
    restoreIpAddresses_scaffold("1921684464", "[192.168.44.64]");
    restoreIpAddresses_scaffold("1921681100", "[19.216.81.100, 192.168.1.100, 192.16.81.100, 192.168.110.0]");
    restoreIpAddresses_scaffold("0000", "[0.0.0.0]");
    restoreIpAddresses_scaffold("101023", "[1.0.10.23, 1.0.102.3, 10.1.0.23, 10.10.2.3, 101.0.2.3]");
    restoreIpAddresses_scaffold("0011255245", "[]");
    TIMER_STOP(restoreIpAddresses);
    util::Log(logESSENTIAL) << "restoreIpAddresses using " << TIMER_MSEC(restoreIpAddresses) << " milliseconds";

    util::Log(logESSENTIAL) << "Running partition tests: ";
    TIMER_START(partition);
    partition_scaffold("aab", "[[a,a,b],[aa,b]]");
    partition_scaffold("a", "[[a]]");
    partition_scaffold("ab", "[[a, b]]");
    TIMER_STOP(partition);
    util::Log(logESSENTIAL) << "partition using " << TIMER_MSEC(partition) << " milliseconds";

    util::Log(logESSENTIAL) << "Running diffWaysToCompute tests: ";
    TIMER_START(diffWaysToCompute);
    diffWaysToCompute_scaffold("2-1-1", "[2,0]");
    diffWaysToCompute_scaffold("2*3-4*5", "[-34,-10,-14,-10,10]");
    TIMER_STOP(diffWaysToCompute);
    util::Log(logESSENTIAL) << "diffWaysToCompute using " << TIMER_MSEC(diffWaysToCompute) << " milliseconds";

    util::Log(logESSENTIAL) << "Running splitIntoFibonacci tests: ";
    TIMER_START(splitIntoFibonacci);
    splitIntoFibonacci_scaffold("123456579", true);
    splitIntoFibonacci_scaffold("11235813", true);
    splitIntoFibonacci_scaffold("112358130", false);
    splitIntoFibonacci_scaffold("0123", false);
    splitIntoFibonacci_scaffold("1101111", true);
    TIMER_STOP(splitIntoFibonacci);
    util::Log(logESSENTIAL) << "splitIntoFibonacci using " << TIMER_MSEC(splitIntoFibonacci) << " milliseconds";

    util::Log(logESSENTIAL) << "Running addOperators tests: ";
    TIMER_START(addOperators);
    addOperators_scaffold("123", 6, "[1+2+3, 1*2*3]");
    addOperators_scaffold("232", 8, "[2+3*2, 2*3+2]");
    addOperators_scaffold("105", 5, "[1*0+5, 10-5]");
    addOperators_scaffold("00", 0, "[0+0, 0-0, 0*0]");
    addOperators_scaffold("3456237490", 9191, "[]");
    TIMER_STOP(addOperators);
    util::Log(logESSENTIAL) << "addOperators using " << TIMER_MSEC(addOperators) << " milliseconds";
}
