#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 997 */

class Solution {
public:
    int findJudge(int N, vector<vector<int>>& trusts);
};

int Solution::findJudge(int N, vector<vector<int>>& trusts) {
/*
    In a town, there are N people labelled from 1 to N.
    There is a rumor that one of these people is secretly the town judge.
    If the town judge exists, then:
        The town judge trusts nobody.
        Everybody (except for the town judge) trusts the town judge.
        There is exactly one person that satisfies properties 1 and 2.
    You are given trust, an array of pairs `trust[i] = [a, b]` representing 
    that the person labelled `a` trusts the person labelled `b`.
    If the town judge exists and can be identified, return the label of the town judge. Otherwise, return -1.
    Hint: node with degree (in_degree - out_degree == N-1) is the judge
*/
    vector<pair<int,int>> graph(N+1); // in_degree, out_degree
    for (auto& t: trusts) {
        graph[t[1]].first++;
        graph[t[0]].second++;
    }
    int judge = -1;
    for (int i=1; i<=N; ++i) {
        if (graph[i].first==N-1 && graph[i].second==0) {
            judge = i;
            break;
        }
    }
    return judge;
}


void findJudge_scaffold(int N, string input, int expectedResult) {
    Solution ss;
    auto trusts = stringTo2DArray<int>(input);
    int actual = ss.findJudge(N, trusts);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", N, input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", N, input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running findJudge tests:");
    TIMER_START(findJudge);
    findJudge_scaffold(2, "[[1,2]]", 2);
    findJudge_scaffold(3, "[[1,3],[2,3]]", 3);
    findJudge_scaffold(3, "[[1,3],[2,3],[3,1]]", -1);
    findJudge_scaffold(3, "[[1,2],[2,3]]", -1);
    findJudge_scaffold(4, "[[1,3],[1,4],[2,3],[2,4],[4,3]]", 3);
    TIMER_STOP(findJudge);
    SPDLOG_WARN("findJudge tests use {} ms", TIMER_MSEC(findJudge));
}
