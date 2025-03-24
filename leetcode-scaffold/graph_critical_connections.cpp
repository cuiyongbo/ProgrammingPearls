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
    // build a undirected graph
    vector<vector<int>> graph(n);
    for (auto& p: connections) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    int t = 0;
    vector<vector<int>> ans;
    vector<int> ts(n, -1); // timestamp when a node is explored for the first time [discovering timestamp]
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
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual:", input1, input2, expectedResult);
        for (auto& s: actual) {
            std::cout << numberVectorToString(s) << std::endl;
        }
    }
}


int main() {
    SPDLOG_WARN("Running criticalConnections tests:");
    TIMER_START(criticalConnections);
    criticalConnections_scaffold(4, "[[0,1],[1,2],[2,0],[1,3]]", "[[1,3]]");
    criticalConnections_scaffold(5, "[[0,1],[1,2],[2,0],[1,3],[3,4]]", "[[1,3],[3,4]]");
    criticalConnections_scaffold(6, "[[0,1],[1,2],[2,0],[1,3],[3,4],[3,5],[4,5]]", "[[1,3]]");
    TIMER_STOP(criticalConnections);
    SPDLOG_WARN("criticalConnections tests use {} ms", TIMER_MSEC(criticalConnections));
}
