#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 698, 93, 131, 241, 282, 842 */

class Solution 
{
public:
    bool canPartitionKSubsets(vector<int>& nums, int k);
    vector<string> restoreIpAddresses(string s);
    vector<vector<string>> partition(string s);
    vector<int> diffWaysToCompute(string input);
    vector<string> addOperators(string num, int target);
};

bool Solution::canPartitionKSubsets(vector<int>& nums, int k)
{
    /*
        Given an array of integers nums and a positive integer k, 
        find whether it’s possible to divide this array into knon-empty 
        subsets whose sums are all equal.

        Example:
            Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
            Output: True
            Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
    */

    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    if(sum % k != 0) return false;

    std::sort(nums.begin(), nums.end());

    int target = sum / k;
    size_t len = nums.size(); 
    vector<bool> used(len, false);
    function<bool(int, int)> dfs = [&](int cur, int left)
    {
        if(left == 0 && std::all_of(used.begin(), used.end(), [](bool v) {return v;})) 
            return true;

        for(size_t i=0; i<len; i++)
        {
            if(used[i]) continue;

            cur += nums[i];

            if(cur > target) break; // pruning branches

            used[i] = true;
            if(cur == target && dfs(0, left-1))
                return true;
            else if(dfs(cur, left))
                return true;
            used[i] = false;

            cur -= nums[i];
        }
        return false;
    };
    return dfs(0, k);
}

vector<string> Solution::restoreIpAddresses(string s)
{
    /*
        Given a string containing only digits, restore it by returning all possible valid IP address combinations.

        Example:

            Input: "25525511135"
            Output: ["255.255.11.135", "255.255.111.35"]
    */

    vector<vector<string>> storage;
    vector<string> courier;
    size_t len = s.size();
    function<void(size_t)> dfs = [&] (size_t pos)
    {
        if(pos == len && (int)courier.size() == 4) 
        {
            storage.push_back(courier);
            return;
        }

        for(size_t j=1; j<4; j++)
        {
            string cur = s.substr(pos, j);
            if(cur.size() != j) break;

            int e = std::stod(cur);
            if(0<=e && e<=255)
            {
                courier.push_back(cur);
                dfs(pos+j);
                courier.pop_back();
            }
        }
    };

    dfs(0);

    vector<string> ans;
    for(const auto& p: storage)
    {
        ans.push_back( p[0] + "." +
                       p[1] + "." + 
                       p[2] + "." + 
                       p[3]);
    }
    return ans;
}

vector<vector<string>> Solution::partition(string s)
{
    /*
        Given a string s, partition s such that every substring of the partition is a palindrome.
        Return all possible palindrome partitioning of s.

        Example:

            Input: "aab"
            Output:
            [
              ["aa","b"],
              ["a","a","b"]
            ]
    */

    auto isValid = [](const string& cur)
    {
        size_t l = 0;
        size_t r = (int)cur.size() - 1;
        while(l < r)
        {
            if(cur[l++] != cur[r--]) 
                return false;
        }
        return true;
    };

    vector<vector<string>> ans;
    vector<string> courier;
    size_t len = s.size();
    function<void(int)> dfs = [&](size_t p)
    {
        if(p == len)
        {
            ans.push_back(courier);
            return;
        }

        for(size_t i=p; i<len; ++i)
        {
            string cur = s.substr(p, i-p+1);
            if(isValid(cur))
            {
                courier.push_back(cur);
                dfs(i+1);
                courier.pop_back();
            }
        }
    };

    dfs(0);
    return ans;
}

vector<int> Solution::diffWaysToCompute(string input)
{
    /*
        Given a string of numbers and operators, return all possible results 
        from computing all the different possible ways to group numbers and operators. 
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

    auto add = [] (int a, int b) { return a+b;};
    auto sub = [] (int a, int b) { return a-b;};
    auto mul = [](int a, int b) { return a*b;};
    map<char, function<int(int, int)>> funcMap;
    funcMap['+'] = add;
    funcMap['-'] = sub;
    funcMap['*'] = mul;

    // memorization
    map<string, vector<int>> sub_solution;
    function<vector<int>(string)> dfs = [&] (string input)
    {
        if(sub_solution.count(input)) return sub_solution[input];

        vector<int> ans;
        bool foundOperator = false;
        for(size_t i=0; i<input.size(); i++)
        {
            if(std::isdigit(input[i])) continue;

            foundOperator = true;
            auto li = input.substr(0, i);
            auto ri = input.substr(i+1);
            const auto& l = dfs(li);
            const auto& r = dfs(ri);

            for(auto a: l)
            {
                for(auto b: r)
                {
                    ans.push_back(funcMap[input[i]](a, b));
                }
            }
        }

        // trivial case: no operator left in input
        if(!foundOperator) ans.push_back(std::stod(input));

        sub_solution[input] = ans;
        return ans;
    };

    return dfs(input);
}

vector<string> Solution::addOperators(string num, int target)
{
    /*
        Given a string that contains only digits 0-9 and a target value, 
        return all possibilities to add binary operators (not unary) +, -, or * 
        between the digits so they evaluate to the target value.

        Examples:

            "123", 6 -> ["1+2+3", "1*2*3"] 
            "232", 8 -> ["2*3+2", "2+3*2"]
            "105", 5 -> ["1*0+5","10-5"]
            "00", 0 -> ["0+0", "0-0", "0*0"]
            "3456237490", 9191 -> []
    */

    auto evaluate = [](vector<string>& exp)
    {
        return 0;
    };

    vector<string> operators {"+", "-", "*"};

    vector<string> exp;
    vector<vector<string>> ans;
    size_t len = num.size();
    function<void(size_t)> backtrace = [&](size_t p)
    {
        if(p == len && evaluate(exp) == target)
        {
            ans.push_back(exp);
            return;
        }

        for(size_t i=p; i<len; i++)
        {
            string cur = num.substr(p, i-p+1);
            exp.push_back(cur);
            for(const auto& s: operators)
            {
                exp.push_back(s);
                backtrace(i+1);
                exp.pop_back();
            }
            exp.pop_back();
        }

    };



}

void canPartitionKSubsets_scaffold(string input1, int input2, bool expectedResult)
{
    Solution ss;
    vector<int> g = stringTo1DArray<int>(input1);
    bool actual = ss.canPartitionKSubsets(g, input2);
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

void restoreIpAddresses_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.restoreIpAddresses(input);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

void partition_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.partition(input);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        for(auto vs: actual)
        {
            util::Log(logERROR) << "*********";
            for(auto s: vs)
            {
                util::Log(logERROR) <<  s;
            }
        }
    }
}

void diffWaysToCompute_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.diffWaysToCompute(input);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << numberVectorToString<int>(actual);
    }
}

void addOperators_scaffold(string input1, int input2, string expectedResult)
{
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.addOperators(input1, input2);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

int main()
{
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
    TIMER_STOP(restoreIpAddresses);
    util::Log(logESSENTIAL) << "restoreIpAddresses using " << TIMER_MSEC(restoreIpAddresses) << " milliseconds";

    util::Log(logESSENTIAL) << "Running partition tests: ";
    TIMER_START(partition);
    partition_scaffold("aab", "[[a,a,b],[aa,b]]");
    TIMER_STOP(partition);
    util::Log(logESSENTIAL) << "partition using " << TIMER_MSEC(partition) << " milliseconds";

    util::Log(logESSENTIAL) << "Running diffWaysToCompute tests: ";
    TIMER_START(diffWaysToCompute);
    diffWaysToCompute_scaffold("2-1-1", "[2,0]");
    diffWaysToCompute_scaffold("2*3-4*5", "[-34,-10,-14,-10,10]");
    TIMER_STOP(diffWaysToCompute);
    util::Log(logESSENTIAL) << "diffWaysToCompute using " << TIMER_MSEC(diffWaysToCompute) << " milliseconds";

    util::Log(logESSENTIAL) << "Running addOperators tests: ";
    TIMER_START(addOperators);
    addOperators_scaffold("123", 6, "[1+2+3, 1*2*3]");
    addOperators_scaffold("232", 8, "[2*3+2, 2+3*2]");
    addOperators_scaffold("105", 5, "[1*0+5, 10-5]");
    addOperators_scaffold("00", 0, "[0+0, 0-0, 0*0]");
    addOperators_scaffold("3456237490", 9191, "[]");
    TIMER_STOP(addOperators);
    util::Log(logESSENTIAL) << "addOperators using " << TIMER_MSEC(addOperators) << " milliseconds";
}