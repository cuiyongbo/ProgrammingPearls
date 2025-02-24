#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 126, 127, 752, 818 */

class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList);
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);
    int openLock(vector<string>& deadends, string target);
    int racecar(int target);
private:
    bool is_valid_transformation(string t, string w) {
        int diff = 0;
        for (int i=0; i<t.size(); ++i) {
            if (t[i] != w[i]) {
                ++diff;
            }
        }
        return diff==1;
    }
};


/*
    Given two words (beginWord and endWord), and a dictionary’s word list, find the length of shortest transformation sequence from beginWord to endWord, 
    Return 0 if there is no such transformation sequence.
    such that:
        Only one letter can be changed at a time.
        Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
    For example, Given:
        beginWord = "hit"
        endWord = "cog"
        wordList = ["hot","dot","dog","lot","log","cog"]
    As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog", return its length 5.
    Note:
        All words have the same length.
        All words contain only lowercase alphabetic characters.
        You may assume no duplicates in the word list.
        You may assume beginWord and endWord are non-empty and are not the same.
    Hint: use bfs(recommended)/dfs to find the shortest path from beginWord to endWord
*/
int Solution::ladderLength(string beginWord, string endWord, vector<string>& wordList) {

{ // a more delicate version of bfs solution
    int steps = 0;
    int sz = wordList.size();
    vector<bool> used(sz, false);
    queue<string> q; q.push(beginWord);
    while (!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; ++i) {
            auto u = q.front(); q.pop();
            if (u == endWord) {
                return steps+1; // add endWord
            }
            for (int v=0; v<sz; ++v) {
                if (used[v]) {
                    continue;
                }
                if (is_valid_transformation(u, wordList[v])) {
                    q.push(wordList[v]);
                    used[v] = true;
                }
            }
        }
        ++steps;
    }
    return 0;
}

{ // naive bfs solution
    queue<string> q; q.push(beginWord);
    set<string> visited; visited.insert(beginWord);
    int steps = 0;
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            if (t == endWord) {
                return steps + 1; // add endWord
            }
            for (auto w: wordList) {
                if (visited.count(w)) {
                    continue;
                }
                if (is_valid_transformation(t, w)) {
                    q.push(w);
                    visited.insert(w);
                }
            }
        }
        ++steps;
    }
    return 0;
}

{ // dfs solution
    // preprocess
    std::sort(wordList.begin(), wordList.end());
    auto it = std::unique(wordList.begin(), wordList.end());
    wordList.erase(it, wordList.end());

    int ans = INT32_MAX;
    int sz = wordList.size();
    vector<bool> used(sz, false);
    vector<string> candidate; candidate.push_back(beginWord);
    function<void(int)> dfs = [&] (int u) {
        if (candidate.back() == endWord) { // reach the destination
#ifdef DEBUG
            std::copy(candidate.begin(), candidate.end(), std::ostream_iterator<string>(cout, ","));
            cout << endl;
#endif
            ans = min(ans, (int)candidate.size());
            return;
        }
        if (u == sz) { // unreachable
            return;
        }
        for (int i=0; i<sz; ++i) {
            if (used[u] || candidate.size() > ans) { // prune useless branches
                continue;
            }
            if (wordList[i] == beginWord) { // avoid loop!!!
                continue;
            }
            if (is_valid_transformation(candidate.back(), wordList[i])) {
                used[i] = true;
                candidate.push_back(wordList[i]);
                dfs(u+1);
                candidate.pop_back();
                used[i] = false;
            }
        }
    };
    dfs(0);
    return ans == INT32_MAX ? 0 : ans;
}

}


/*
    Given two words (beginWord and endWord), and a dictionary’s word list, 
    find all shortest transformation sequence(s) from beginWord to endWord, such that:
        Only one letter can be changed at a time
        Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
    Return an empty list if there is no such transformation sequence.
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
        All words have the same length.
        All words contain only lowercase alphabetic characters.
        You may assume no duplicates in the word list.
        You may assume beginWord and endWord are non-empty and are not the same.
*/
vector<vector<string>> Solution::findLadders(string beginWord, string endWord, vector<string>& wordList) {
    vector<vector<string>> ans;
    // we need to find the best_path_len first before performing dfs search, otherwise dfs would spend too much time on invalid path
    int best_path_len = ladderLength(beginWord, endWord, wordList);
    vector<string> path; path.push_back(beginWord); // initialize path, visited with start node
    set<string> visited; visited.insert(beginWord);
    function<void(string)> backtrace = [&] (string u) {
        //SPDLOG_INFO("{}, depth: {}, best_path_len: {}", u, path.size(), best_path_len);
        // prune invalid branches
        if (path.size() > best_path_len) {
            return;
        }
        if (u == endWord) {
            if (path.size() < best_path_len) {
                ans.clear();
                best_path_len = path.size();
            }
            // DEBUG
            ans.push_back(path);
            return;
        }
        for (int i=0; i<wordList.size(); i++) {
            if (visited.count(wordList[i])) {
                continue;
            }
            // prune invalid branches
            if (is_valid_transformation(u, wordList[i])) {
                visited.insert(wordList[i]);
                path.push_back(wordList[i]);
                backtrace(wordList[i]);
                path.pop_back();
                visited.erase(wordList[i]);
            }
        }
    };
    backtrace(beginWord);
    return ans;
}


int Solution::openLock(vector<string>& deadends, string target) {
/*
    You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. 
    The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. **Each move consists of turning one wheel one slot.**
    The lock initially starts at '0000', a string representing the state of the 4 wheels.
    You are given a list of deadends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.
    Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.
    Note:
        The length of deadends will be in the range [1, 500].
        target will not be in the list deadends.
        Every string in deadends and the string target will be a string of 4 digits from the 10,000 possibilities '0000' to '9999'.
    
    Hint: BFS or Bidirectional BFS
*/

if (0) { // naive bfs
    int steps = 0;
    string start("0000");
    set<string> forbidden(deadends.begin(), deadends.end());
    if (forbidden.count(start) || forbidden.count(target)) {
        return -1;
    }
    queue<string> q; q.push(start);
    set<string> visited; visited.insert(start);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; k--) {
            auto u = q.front(); q.pop();
            if (u == target) {
                return steps;
            }
            for (int i=0; i<4; i++) {
                for (auto d: {-1, 1}) {
                    char c = u[i];
                    u[i] = u[i] + d;
                    if (u[i] < '0') {
                        u[i] = '9';
                    } else if (u[i] > '9') {
                        u[i] = '0';
                    }
                    if (forbidden.count(u)==0 && visited.count(u)==0) {
                        q.push(u);
                        visited.insert(u);
                    }
                    u[i] = c; // IMPORTANT: restore u[i]
                }
            }
        }
        steps++;
    }
    return -1;
}

{ // bidirectional bfs
    string start("0000");
    set<string> forbidden(deadends.begin(), deadends.end());
    if (forbidden.count(start) != 0 || forbidden.count(target) != 0) {
        return -1;
    }
    int forward_steps = 0;
    queue<string> q1; q1.push(start);
    set<string> v1; v1.emplace(start);
    int backward_steps = 0;
    queue<string> q2; q2.push(target);
    set<string> v2; v2.emplace(target);
    auto routeStep = [&] (queue<string>& q, set<string>& visited, set<string>& reverse_visited) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            if (reverse_visited.count(u)) {
                return true;
            }
            for (int i=0; i<4; i++) {
                for (auto d: {-1, 1}) {
                    char c = u[i];
                    u[i] = u[i] + d;
                    if (u[i] < '0') {
                        u[i] = '9';
                    } else if (u[i] > '9') {
                        u[i] = '0';
                    }
                    if (forbidden.count(u)==0 && visited.count(u)==0) {
                        q.push(u);
                        visited.insert(u);
                    }
                    u[i] = c; // IMPORTANT: restore u[i]
                }
            }
        }
        return false;
    };
    while (!q1.empty() && !q2.empty()) {
        forward_steps++;
        if (routeStep(q1, v1, v2)) {
            return forward_steps + backward_steps -1;
        }
        backward_steps++;
        if (routeStep(q2, v2, v1)) {
            return forward_steps + backward_steps - 1;
        }
    }
    return -1;
}

}


/*
    Your car starts at position 0 and speed=1 on an infinite number line. (Your car can go into negative positions.)
    Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse).
    When you get an instruction “A”, your car does the following: position += speed, speed *= 2.
    When you get an instruction “R”, your car does the following: if your speed is positive then speed = -1 , otherwise speed = 1. (Your position stays the same.)
    For example, after commands “AAR”, your car goes to positions 0->1->3->3, and your speed goes to 1->2->4->-1.
    Now for some target position, say the length of the shortest sequence of instructions to get there.
*/
int Solution::racecar(int target) {

{ // bfs solution
    int steps = 0;
    using element_t = std::pair<int, int>; // position, speed
    queue<element_t> q; q.emplace(0, 1);
    set<element_t> visited; visited.emplace(0, 1);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; k--) {
            auto t = q.front(); q.pop();
            if (t.first == target) {
                return steps;
            }
            element_t p;
            // accelerate
            // we won't choose acceleration if we would surpassed target after accelerating next time
            p = std::make_pair(t.first+t.second, t.second*2);
            if (p.first <= 2*target && visited.count(p) == 0) {
                q.push(p);
                visited.insert(p);
            }
            // reverse
            p = std::make_pair(t.first, t.second>0 ? -1 : 1);
            if (visited.count(p) == 0) {
                q.push(p);
                visited.insert(p);
            }
        }
        steps++;
    }
    return -1;
}

{ // dp solution
    // dp[i] means minimum commands to reach position i
    vector<int> dp(target+1, 0);
    function<int(int)> helper = [&](int t) {
        if (dp[t] > 0) {
            return dp[t];
        }
        // AA...A (nA) best case
        int n = std::ceil(std::log2(t+1));
        if ((1<<n) == t+1) {
            dp[t] = n;
            return dp[t];
        }
        // AA...AR (nA + R) + dp(left) 
        dp[t] = n+1 + helper((1<<n)-t-1);
        for (int m=0; m<n-1; ++m) {
            int cur = (1<<(n-1)) - (1<<m);
            // AA...ARA...AR (n-1A + 1R + mA + 1R) + dp(left) 
            dp[t] = std::min(dp[t], n+m+1+helper(t-cur));
        }
        return dp[t];
    };
    return helper(target);
}

}

void ladderLength_scaffold(string input1, string input2, string input3, int expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input3);
    int actual = ss.ladderLength(input1, input2, dict);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: ", input1, input2, expectedResult, actual);
    }
}

void findLadders_scaffold(string input1, string input2, string input3, string expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input3);
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.findLadders(input1, input2, dict);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual:", input1, input2, input3, expectedResult);
        for (const auto& path: actual) {
            for (int i=0; i<(int)path.size(); ++i) {
                if (i != 0) cout << " -> ";
                cout << path[i];
            }
            cout << endl;
        }
    }
}


void openLock_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<string> deadends = stringTo1DArray<string>(input1);
    int actual = ss.openLock(deadends, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


void racecar_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.racecar(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running ladderLength tests:");
    TIMER_START(ladderLength);
    ladderLength_scaffold("hit", "clog", "[hot,dot,dog,lot,log,cog]", 0);
    ladderLength_scaffold("hit", "cog", "[hot,dot,dog,lot,log,cog]", 5);
    ladderLength_scaffold("cot", "log", "[hot,dot,dog,lot,log,cog]", 3);
    ladderLength_scaffold("cot", "lot", "[hot,dot,dog,lot,log,cog]", 2);
    ladderLength_scaffold("qa", "sq", "[si,go,se,cm,so,ph,mt,db,mb,sb,kr,ln,tm,le,av,sm,ar,ci,ca,br,ti,ba,to,ra,fa,yo,ow,sn,ya,cr,po,fe,ho,ma,re,or,rn,au,ur,rh,sr,tc,lt,lo,as,fr,nb,yb,if,pb,ge,th,pm,rb,sh,co,ga,li,ha,hz,no,bi,di,hi,qa,pi,os,uh,wm,an,me,mo,na,la,st,er,sc,ne,mn,mi,am,ex,pt,io,be,fm,ta,tb,ni,mr,pa,he,lr,sq,ye]", 5);
    TIMER_STOP(ladderLength);
    SPDLOG_WARN("ladderLength using {} ms", TIMER_MSEC(ladderLength));

    SPDLOG_WARN("Running findLadders tests:");
    TIMER_START(findLadders);
    findLadders_scaffold("hit", "clog", "[hot,dot,dog,lot,log,cog]", "[]");
    findLadders_scaffold("hit", "cog", "[hot,dot,dog,lot,log,cog]", "[[hit,hot,dot,dog,cog], [hit,hot,lot,log,cog]]");
    findLadders_scaffold("cot", "lot", "[hot,dot,dog,lot,log,cog]", "[[cot, lot]]");
    findLadders_scaffold("qa", "sq", "[si,go,se,cm,so,ph,mt,db,mb,sb,kr,ln,tm,le,av,sm,ar,ci,ca,br,ti,ba,to,ra,fa,yo,ow,sn,ya,cr,po,fe,ho,ma,re,or,rn,au,ur,rh,sr,tc,lt,lo,as,fr,nb,yb,if,pb,ge,th,pm,rb,sh,co,ga,li,ha,hz,no,bi,di,hi,qa,pi,os,uh,wm,an,me,mo,na,la,st,er,sc,ne,mn,mi,am,ex,pt,io,be,fm,ta,tb,ni,mr,pa,he,lr,sq,ye]", "[[qa,ba,be,se,sq],[qa,ba,bi,si,sq],[qa,ba,br,sr,sq],[qa,ca,ci,si,sq],[qa,ca,cm,sm,sq],[qa,ca,co,so,sq],[qa,ca,cr,sr,sq],[qa,fa,fe,se,sq],[qa,fa,fm,sm,sq],[qa,fa,fr,sr,sq],[qa,ga,ge,se,sq],[qa,ga,go,so,sq],[qa,ha,he,se,sq],[qa,ha,hi,si,sq],[qa,ha,ho,so,sq],[qa,la,le,se,sq],[qa,la,li,si,sq],[qa,la,ln,sn,sq],[qa,la,lo,so,sq],[qa,la,lr,sr,sq],[qa,la,lt,st,sq],[qa,ma,mb,sb,sq],[qa,ma,me,se,sq],[qa,ma,mi,si,sq],[qa,ma,mn,sn,sq],[qa,ma,mo,so,sq],[qa,ma,mr,sr,sq],[qa,ma,mt,st,sq],[qa,na,nb,sb,sq],[qa,na,ne,se,sq],[qa,na,ni,si,sq],[qa,na,no,so,sq],[qa,pa,pb,sb,sq],[qa,pa,ph,sh,sq],[qa,pa,pi,si,sq],[qa,pa,pm,sm,sq],[qa,pa,po,so,sq],[qa,pa,pt,st,sq],[qa,ra,rb,sb,sq],[qa,ra,re,se,sq],[qa,ra,rh,sh,sq],[qa,ra,rn,sn,sq],[qa,ta,tb,sb,sq],[qa,ta,tc,sc,sq],[qa,ta,th,sh,sq],[qa,ta,ti,si,sq],[qa,ta,tm,sm,sq],[qa,ta,to,so,sq],[qa,ya,yb,sb,sq],[qa,ya,ye,se,sq],[qa,ya,yo,so,sq]]");
    TIMER_STOP(findLadders);
    SPDLOG_WARN("findLadders using {} ms", TIMER_MSEC(findLadders));

    SPDLOG_WARN("Running openLock tests:");
    TIMER_START(openLock);
    openLock_scaffold("[0201,0101,0102,1212,2002]", "0202", 6);
    openLock_scaffold("[8888]", "0009", 1);
    openLock_scaffold("[8887,8889,8878,8898,8788,8988,7888,9888]", "8888", -1);
    openLock_scaffold("[0000]", "8888", -1);
    TIMER_STOP(openLock);
    SPDLOG_WARN("openLock using {} ms", TIMER_MSEC(openLock));

    SPDLOG_WARN("Running racecar tests:");
    TIMER_START(racecar);
    racecar_scaffold(3, 2);
    racecar_scaffold(6, 5);
    racecar_scaffold(100, 19);
    TIMER_STOP(racecar);
    SPDLOG_WARN("racecar using {} ms", TIMER_MSEC(racecar));

    return 0;
}
