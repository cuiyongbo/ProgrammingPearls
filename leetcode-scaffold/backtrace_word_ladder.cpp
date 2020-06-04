#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 126, 127, 752, 818 */

class Solution 
{
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList);
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);
    int openLock(vector<string>& deadends, string target);
    int racecar(int target);
    int racecar_dp(int target);
};

int Solution::ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
    /*
        Given two words (beginWord and endWord), and a dictionary’s word list, 
        find the length of shortest transformation sequence from beginWord to endWord, 
        such that:
            Only one letter can be changed at a time.
            Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
        For example, Given:
            beginWord = "hit"
            endWord = "cog"
            wordList = ["hot","dot","dog","lot","log","cog"]
        As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
        return its length 5.

        Note:
            Return 0 if there is no such transformation sequence.
            All words have the same length.
            All words contain only lowercase alphabetic characters.
            You may assume no duplicates in the word list.
            You may assume beginWord and endWord are non-empty and are not the same.
    */

    int len = (int)beginWord.size();
    auto isValid = [&](const string& u, const string& v)
    {
        int count = 0;
        for(int i=0; i<len; ++i)
        {
            if(u[i] != v[i]) ++count;
        }
        return count == 1;
    };

    int steps = 0;
    unordered_set<string> visited;
    visited.emplace(beginWord);
    queue<string> q;
    q.push(beginWord);
    while(!q.empty())
    {
        int size = (int)q.size();
        while(size-- != 0)
        {
            auto u = q.front(); q.pop();
            if(u == endWord) return steps;
            for(const auto& v: wordList)
            {
                if(visited.count(v) != 0) continue;
                if(isValid(u, v)) 
                {
                    q.push(v);
                    visited.emplace(v);
                }
            }
        }
        ++steps;
    }
    return 0;
}

vector<vector<string>> Solution::findLadders(string beginWord, string endWord, vector<string>& wordList)
{
    /*
        Given two words (beginWord and endWord), and a dictionary’s word list, 
        find all shortest transformation sequence(s) from beginWord to endWord, such that:

            Only one letter can be changed at a time
            Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
        For example, Given:
            beginWord = "hit"
            endWord = "cog"
            wordList = ["hot","dot","dog","lot","log","cog"]
        Return
          [
            ["hit","hot","dot","dog","cog"],
            ["hit","hot","lot","log","cog"]
          ]
        
        Note:
            Return an empty list if there is no such transformation sequence.
            All words have the same length.
            All words contain only lowercase alphabetic characters.
            You may assume no duplicates in the word list.
            You may assume beginWord and endWord are non-empty and are not the same.
    */

    vector<vector<string>> ans;
    unordered_map<string, vector<string>> graph;

    int len = (int)beginWord.size();
    auto isValid = [&](const string& u, const string& v)
    {
        int count = 0;
        for(int i=0; i<len; ++i)
        {
            if(u[i] != v[i]) ++count;
        }
        return count == 1;
    };

    size_t path_len = 0;
    auto buildGraph = [&]()
    {
        size_t steps = 0;
        queue<string> q; q.push(beginWord);
        unordered_set<string> visited; visited.emplace(beginWord);
        while(!q.empty())
        {
            ++steps;
            int size = (int)q.size();
            while(size-- != 0)
            {
                auto u = q.front(); q.pop();
                if(u == endWord) 
                {
                    path_len = steps;
                    continue;
                }
                for(const auto& v: wordList)
                {
                    if(isValid(u, v)) 
                    {
                        graph[u].push_back(v);
                        if(visited.count(v) == 0)
                        {
                            visited.emplace(v);
                            q.push(v);
                        }
                    }
                }
            }
        }
    };

    buildGraph();
    if(path_len == 0) return ans;

    // not necessary, out of convience for test
    for(auto& it: graph)
    {
        std::sort(it.second.begin(), it.second.end());
    }

    vector<string> path;
    path.push_back(beginWord);
    unordered_set<string> visited; visited.emplace(beginWord);
    function<void(const string&)> backtrace = [&](const string& u)
    {
        if(u == endWord)
        {
            ans.push_back(path);
            return;
        }

        for(const auto& v: graph[u])
        {
            if(visited.count(v) == 0 && path.size() != path_len)
            {
                visited.emplace(v);
                path.push_back(v);
                backtrace(v);
                path.pop_back();
                visited.erase(v);
            }
        }
    };

    backtrace(beginWord);
    return ans;
}

int Solution::openLock(vector<string>& deadends, string target)
{
    /*
        You have a lock in front of you with 4 circular wheels. 
        Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. 
        The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', 
        or '0' to be '9'. Each move consists of turning one wheel one slot.

        The lock initially starts at '0000', a string representing the state of the 4 wheels.

        You are given a list of deadends dead ends, meaning if the lock displays any of these codes, 
        the wheels of the lock will stop turning and you will be unable to open it.

        Given a target representing the value of the wheels that will unlock the lock, 
        return the minimum total number of turns required to open the lock, or -1 if it is impossible.

        Note:
            The length of deadends will be in the range [1, 500].
            target will not be in the list deadends.
            Every string in deadends and the string target will be a string of 4 digits from the 10,000 possibilities '0000' to '9999'.
        
        Hint: Bidirectional BFS
    */

    const string start = "0000";
    unordered_set<string> forbidden(deadends.begin(), deadends.end());
    if(forbidden.count(start) != 0) return -1;

    unordered_set<string> v1; v1.emplace(start);
    unordered_set<string> v2; v2.emplace(target);
    queue<string> q1; q1.push(start);
    queue<string> q2; q2.push(target);
    int s1=0, s2=0;

    bool meet = false;    
    auto routingStep = [&](queue<string>& q, unordered_set<string>& visited, unordered_set<string>& rev_visited, int& steps)
    {
        for(size_t i=q.size(); i != 0; --i)
        {
            auto u = q.front(); q.pop(); 
            auto v = u;
            for(int k=0; k<4; ++k)
            {
                for(int j=-1; j<2; j+=2)
                {
                    v[k] = (u[k]-'0' + 10 + j) % 10 + '0';
                    if(visited.count(v) == 0 && forbidden.count(v) == 0)
                    {
                        if(rev_visited.count(v) != 0)
                        {
                            meet = true;
                            return;
                        }
                        visited.emplace(v);
                        q.push(v);
                    }
                    v[k] = u[k];
                }
            }
        }
        ++steps;
    };

    while(!q1.empty() && !q2.empty())
    {
        routingStep(q1, v1, v2, s1);
        if(meet) return s1+s2+1;

        routingStep(q2, v2, v1, s2);
        if(meet) return s1+s2+1;
    }

    return -1;
}

namespace std
{
    template<>
    struct hash<pair<int, int>>
    {
        size_t operator()(const pair<int, int>& p) const
        {
            return (((size_t)p.first) << 32) | p.second;
        }
    };
}

int Solution::racecar(int target) 
{    
    /*
        Your car starts at position 0 and speed +1 on an infinite number line.  (Your car can go into negative positions.)
        Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse).
        When you get an instruction “A”, your car does the following: position += speed, speed *= 2.
        When you get an instruction “R”, your car does the following: if your speed is positive then speed = -1 , 
        otherwise speed = 1.  (Your position stays the same.)
        For example, after commands “AAR”, your car goes to positions 0->1->3->3, and your speed goes to 1->2->4->-1.
        Now for some target position, say the length of the shortest sequence of instructions to get there.
    */

    queue<pair<int, int>> q;
    q.push({0, 1});
    unordered_set<pair<int, int>> visited; // only store state for command R
    visited.insert({0, 1});
    visited.insert({0, -1});
    int steps = 0;
    while (!q.empty()) 
    {
        for(size_t i=q.size(); i != 0; --i)
        {
            auto p = q.front(); q.pop();

            // command A
            {            
                auto p1 = std::make_pair(p.first + p.second, p.second*2);
                if(p1.first == target) return steps+1;
                if(p1.first>0 && p1.first < 2* target) q.push(p1);
            }

            // command R
            {
                int s = p.second > 0 ? -1 : 1;
                auto p2 = std::make_pair(p.first, s);
                if(visited.count(p2) == 0)
                {
                    q.push(p2);
                    visited.insert(p2);
                }
            }
        }
        ++steps;
    }
    return -1;
}

int Solution::racecar_dp(int target)
{
    // dp[i] means minimum commands to reach position i
    vector<int> dp(target+1, 0);
    function<int(int)> helper = [&](int t)
    {
        if(dp[t] > 0) return dp[t];
        int n = std::ceil(std::log2(t+1));

        // AA...A (nA) best case
        if((1<<n) == t+1) return dp[t] = n;

        // AA...AR (nA + 1R) + dp(left) 
        dp[t] = n+1 + helper((1<<n)-t-1);
        for(int m=0; m<n-1; ++m)
        {
            int cur = (1<<(n-1)) - (1<<m);

            // AA...ARA...AR (n-1A + 1R + mA + 1R) + dp(left) 
            dp[t] = std::min(dp[t], n+m+1+helper(t-cur));
        }
        return dp[t];
    };

    return helper(target);
}

void ladderLength_scaffold(string input1, string input2, string input3, int expectedResult)
{
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input3);
    int actual = ss.ladderLength(input1, input2, dict);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findLadders_scaffold(string input1, string input2, string input3, string expectedResult)
{
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input3);
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.findLadders(input1, input2, dict);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& path: actual)
        {
            for(int i=0; i<(int)path.size(); ++i)
            {
                if(i != 0) cout << " -> ";
                cout << path[i];
            }
            cout << endl;
        }
    }
}

void openLock_scaffold(string input1, string input2, int expectedResult)
{
    Solution ss;
    vector<string> deadends = stringTo1DArray<string>(input1);
    int actual = ss.openLock(deadends, input2);
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

void racecar_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.racecar(input);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running ladderLength tests: ";
    TIMER_START(ladderLength);
    ladderLength_scaffold("hit", "clog", "[hot,dot,dog,lot,log,cog]", 0);
    ladderLength_scaffold("hit", "cog", "[hot,dot,dog,lot,log,cog]", 4);
    TIMER_STOP(ladderLength);
    util::Log(logESSENTIAL) << "ladderLength using " << TIMER_MSEC(ladderLength) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running findLadders tests: ";
    TIMER_START(findLadders);
    findLadders_scaffold("hit", "clog", "[hot,dot,dog,lot,log,cog]", "[]");
    findLadders_scaffold("hit", "cog", "[hot,dot,dog,lot,log,cog]", "[[hit,hot,dot,dog,cog], [hit,hot,lot,log,cog]]");
    findLadders_scaffold("cot", "lot", "[hot,dot,dog,lot,log,cog]", "[[cot, lot]]");
    TIMER_STOP(findLadders);
    util::Log(logESSENTIAL) << "findLadders using " << TIMER_MSEC(findLadders) << " milliseconds";

    util::Log(logESSENTIAL) << "Running openLock tests: ";
    TIMER_START(openLock);
    openLock_scaffold("[0201,0101,0102,1212,2002]", "0202", 6);
    openLock_scaffold("[8888]", "0009", 1);
    openLock_scaffold("[8887,8889,8878,8898,8788,8988,7888,9888]", "8888", -1);
    openLock_scaffold("[0000]", "8888", -1);
    TIMER_STOP(openLock);
    util::Log(logESSENTIAL) << "openLock using " << TIMER_MSEC(openLock) << " milliseconds";

    util::Log(logESSENTIAL) << "Running racecar tests: ";
    TIMER_START(racecar);
    racecar_scaffold(3, 2);
    racecar_scaffold(6, 5);
    racecar_scaffold(100, 19);
    TIMER_STOP(racecar);
    util::Log(logESSENTIAL) << "racecar using " << TIMER_MSEC(racecar) << " milliseconds";

    Solution ss;
    vector<int> vi;
    generateTestArray(vi, 100, false, false);
    for(int i=0; i<(int)vi.size(); ++i)
    {
        int s1 = ss.racecar(vi[i]);
        int s2 = ss.racecar_dp(vi[i]);
        if(s1 != s2)
        {
            util::Log(logERROR) << "Test " << i << ", Target: " << vi[i] << ", racecar: " << s1 << ", racecar_dp: " << s2;
        }
    }
}
