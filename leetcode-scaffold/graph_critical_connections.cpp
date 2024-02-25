#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 1192*/

class Solution {
public:
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections);
};


/*
    There are n servers numbered from 0 to n-1 connected by undirected server-to-server connections forming a network where connections[i] = [a, b] 
    represents a connection between servers a and b. Any server can reach any other server directly or indirectly through the network.
    A critical connection is a connection that, if removed, will make some server unable to reach some other server.
    Return all critical connections in the network in sorted order.
    Note:
        connections[i][0] != connections[i][1]
        There are no repeated connections.
    Hint: tarjan algorithm
*/
vector<vector<int>> Solution::criticalConnections(int n, vector<vector<int>>& connections) {
    vector<vector<int>> graph(n);
    for (auto& p: connections) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    int t = 0;
    vector<vector<int>> ans;
    vector<int> ts(n, -1); // timestamp when a node is explored for the first time
    // return the earliest timestamp of node(s) reachable from u, including u itself
    function<int(int, int)> tarjan = [&](int u, int parent) {
        int min_u = ts[u] = t++; 
        for (auto& v: graph[u]) {
            if (ts[v] == -1) { // node v is unvisited yet
                int min_v = tarjan(v, u);
                min_u = min(min_u, min_v);
                if (ts[u] < min_v) {
                    ans.push_back({u, v});
                }
            } else if (v != parent) {
                min_u = min(min_u, ts[v]);
            }
        }
        return min_u;
    };

    tarjan(0, -1);

    for (auto& v: ans) {
        if (v[0] > v[1]) {
            std::swap(v[0], v[1]);
        }
    }
    std::sort(ans.begin(), ans.end());
    return ans;
}

void criticalConnections_scaffold(int input1, string input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> connections = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.criticalConnections(input1, connections);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expected: " << expectedResult << ") failed, actual: ";
        for (auto& s: actual) {
            util::Log(logERROR) << numberVectorToString(s);
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running criticalConnections tests:";
    TIMER_START(criticalConnections);
    criticalConnections_scaffold(4, "[[0,1],[1,2],[2,0],[1,3]]", "[[1,3]]");
    TIMER_STOP(criticalConnections);
    util::Log(logESSENTIAL) << "criticalConnections using " << TIMER_MSEC(criticalConnections) << " milliseconds";
}
