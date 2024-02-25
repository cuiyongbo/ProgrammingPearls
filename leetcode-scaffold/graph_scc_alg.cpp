#include "leetcode.h"

using namespace std;
using namespace osrm;

enum Algorithm {
    Algorithm_tarjan,
    Algorithm_kosaraju
};

const char* Algorithm_toString(Algorithm type) {
    switch(type) {
        case Algorithm_tarjan:
            return "tarjan";
        case Algorithm_kosaraju:
            return "kosaraju";
        default:
            return "unknown";
    }
}


/*
    Given a directed graph with nodes labelled 1 to `node_count`, edge `[u, v]`
    means there is a directed edge connected node u and v, where u is a parent of child v.
    find SCCs in a directed graph.
    Hint: tarjan algorithm or kosaraju algorithm
*/
class Solution {
public:
    vector<vector<int>> tarjan_alg(int n, const vector<vector<int>>& edges);
    vector<vector<int>> kosaraju_alg(int n, const vector<vector<int>>& edges);
};


vector<vector<int>> Solution::tarjan_alg(int node_count, const vector<vector<int>>& edges) {
    vector<vector<int>> graph(node_count);
    for (const auto& p: edges) {
        graph[p[0]-1].push_back(p[1]-1);
    }

    vector<int> discovered_time(node_count, -1);
    vector<int> low_time(node_count, -1);
    vector<bool> on_stack(node_count, false);

    int tick = 0;
    stack<int> st;
    vector<vector<int>> scc_list;

    function<void(int)> dfs = [&](int u) {
        discovered_time[u] = tick;
        low_time[u] = tick;
        st.push(u);
        on_stack[u] = true;
        tick++;

        for (const auto& v: graph[u]) {
            if (discovered_time[v] == -1)
            {
                // unvisited
                dfs(v);
                low_time[u] = std::min(low_time[u], low_time[v]);
            } else if (on_stack[v]) {
                // already discovered
                low_time[u] = std::min(low_time[u], discovered_time[v]);
            }
        } 

        if (discovered_time[u] == low_time[u]) {
            int v = u;
            vector<int> scc;
            do
            {
                v = st.top();
                st.pop();
                on_stack[v] = false;
                scc.push_back(v+1);
            } while (u != v);
            scc_list.push_back(scc);            
        }
    };

    for (int u=0; u < node_count; ++u) {
        if (discovered_time[u] == -1) {
            dfs(u);
        }
    }

    for (auto& scc: scc_list) std::sort(scc.begin(), scc.end());
    std::sort(scc_list.begin(), scc_list.end());

    return scc_list;
}


vector<vector<int>> Solution::kosaraju_alg(int node_count, const vector<vector<int>>& edges) {
    vector<vector<int>> graph(node_count+1);
    vector<vector<int>> reverse_graph(node_count+1);
    for (auto& e: edges) {
        graph[e[0]].push_back(e[1]);
        reverse_graph[e[1]].push_back(e[0]);
    }

    stack<int> st;
    set<int> visited;
    function<void(int)> forward_dfs = [&] (int u) {
        visited.insert(u);
        for (auto v: graph[u]) {
            if (visited.count(v) == 0) {
                forward_dfs(v);
            }
        }
        st.push(u);
    };
    for (int u=1; u<node_count; ++u) {
        if (visited.count(u)==0) {
            forward_dfs(u);
        }
    }

    vector<int> scc;
    function<void(int)> reverse_dfs = [&] (int u) {
        scc.push_back(u);
        visited.insert(u);
        for (auto v: reverse_graph[u]) {
            if (visited.count(v) == 0) {
                reverse_dfs(v);
            }
        }
    };

    visited.clear();
    vector<vector<int>> ans;
    while (!st.empty()) {
        auto u = st.top(); st.pop();
        if (visited.count(u) == 0) {
            reverse_dfs(u);
            std::sort(scc.begin(), scc.end());
            ans.push_back(scc);
            scc.clear();
        }
    }
    return ans;
}


void scc_alg_scaffold(int input1, string input2, string expectedResult, Algorithm type) {
    Solution ss;
    vector<vector<int>> edges = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);

    vector<vector<int>> actual;
    if (type == Algorithm_tarjan) {
        actual = ss.tarjan_alg(input1, edges);
    } else if (type == Algorithm_kosaraju) {
        actual = ss.kosaraju_alg(input1, edges);
    } else {
        util::Log(logERROR) << "unknown algorithm";
        return;
    }

    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 
                                            << ", expectedResult: " << expectedResult 
                                            << ", " << Algorithm_toString(type) << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 
                                            << ", expectedResult: " << expectedResult 
                                            << ", " << Algorithm_toString(type) << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running tarjan_alg tests:";
    TIMER_START(tarjan_alg);
    scc_alg_scaffold(8, "[[1,3],[2,1],[3,2],[3,4],[4,5],[5,6],[6,7],[7,8],[8,5]]", "[[1,2,3],[4],[5,6,7,8]]", Algorithm_tarjan);
    scc_alg_scaffold(9, "[[1,2],[2,1],[2,5],[3,4],[4,3],[4,5],[5,6],[6,7],[7,8],[8,6],[8,9]]", "[[1,2],[3,4],[5],[6,7,8],[9]]", Algorithm_tarjan);
    scc_alg_scaffold(7, "[[1, 2],[2, 3],[3, 4],[4, 1],[2, 5],[5, 6],[5, 7],[7, 6]]", "[[1,2,3,4],[5],[6],[7]]", Algorithm_tarjan);
    TIMER_STOP(tarjan_alg);
    util::Log(logESSENTIAL) << "tarjan_alg using " << TIMER_MSEC(tarjan_alg) << " milliseconds";

    util::Log(logESSENTIAL) << "Running kosaraju_alg tests:";
    TIMER_START(kosaraju_alg);
    scc_alg_scaffold(8, "[[1,3],[2,1],[3,2],[3,4],[4,5],[5,6],[6,7],[7,8],[8,5]]", "[[1,2,3],[4],[5,6,7,8]]", Algorithm_kosaraju);
    scc_alg_scaffold(9, "[[1,2],[2,1],[2,5],[3,4],[4,3],[4,5],[5,6],[6,7],[7,8],[8,6],[8,9]]", "[[1,2],[3,4],[5],[6,7,8],[9]]", Algorithm_kosaraju);
    scc_alg_scaffold(7, "[[1, 2],[2, 3],[3, 4],[4, 1],[2, 5],[5, 6],[5, 7],[7, 6]]", "[[1,2,3,4],[5],[6],[7]]", Algorithm_kosaraju);
    TIMER_STOP(kosaraju_alg);
    util::Log(logESSENTIAL) << "kosaraju_alg using " << TIMER_MSEC(kosaraju_alg) << " milliseconds";
}
