#include "leetcode.h"

using namespace std;

/* leetcode: 684, 685, 1319 */
class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges);
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges);
    int makeConnected(int n, vector<vector<int>>& connections);
};


/*
    In this problem, a tree is an *undirected* graph that is connected and has no cycles.

    You are given a graph that started as a tree with n nodes labelled from 1 to n, with one additional edge added. 
    The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. 
    The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.

    Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.

    Constraints:
        N == edges.length
        edges[i].length == 2
        1 <= ai < bi <= edges.length
        ai != bi
        There are no repeated edges.
        The given graph is connected.
*/
vector<int> Solution::findRedundantConnection(vector<vector<int>>& edges) {
    vector<int> ans;
    DisjointSet dsu(edges.size());
    for (auto& e: edges) {
        // return the last edge which results in a cycle in the graph
        if (!dsu.unionFunc(e[0], e[1])) {
            ans = e;
        }
    }
    return ans;
}


/*
    In this problem, a rooted tree is a directed graph such that, there is exactly one node (the root) for which all other nodes are descendants of this node,
    plus every node has exactly one parent, except for the root node which has no parents.

    The given input is a directed graph that started as a rooted tree with N nodes (with distinct values 1, 2, …, N),
    with one additional directed edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

    The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] that represents a directed edge connecting nodes u and v, where u is a parent of child v.
    Return an edge that can be removed so that the resulting graph is a rooted tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.

    Constraints:
        N == edges.length (note this constraint)
        edges[i].length == 2
        1 <= u, v <= N
        u != v
    Hint:

case 1: the additional directed edge points to root node, which results in a cycle, and in-degree of all nodes is equal to 1, revert to findRedundantConnection

```plaintext
    1 (root)
    /^
    v  \
    2-->3
```

case 2: the additional directed edge points to non-root node, and the in-degree of that node becomes 2, but there is no cycle in the graph

```plaintext
    1 (root)
    / \
    v   v
    2-->3
```

case 3: the additional directed edge points to non-root node, and the in-degree of that node becomes 2, but there is a cycle in the graph

```plaintext
    1
    |
    v
    2 <--> 3
```
*/
vector<int> Solution::findRedundantDirectedConnection(vector<vector<int>>& edges) {
    // 1. test whether the additional edge points to root node or not
    bool point_to_root = true;
    int node_with_2_indegree = -1;
    int N = edges.size();
    vector<int> indegrees(N+1, 0); // NOTE that index 0 is not used
    vector<vector<int>> graph(N+1);
    for (auto e: edges) {
        graph[e[0]].push_back(e[1]);
        indegrees[e[1]]++;
        if (indegrees[e[1]] == 2) {
            point_to_root = false;
            node_with_2_indegree = e[1];
        }
    }
    // case 1
    if (point_to_root) {
        vector<int> ans;
        DisjointSet dsu(N);
        for (auto e: edges) {
            if (!dsu.unionFunc(e[0], e[1])) {
                ans = e;
            }
        }
        return ans;
    }
    // 2 test if there is a cycle
    // return false if there is a cycle in the graph
    vector<int> visited(N+1, 0);
    std::function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                if (!dfs(v)) {
                    return false;
                }
            } else if (visited[v] == 1) {
                return false;
            }
        }
        visited[u] = 2;
        return true;
    };
    int root_node = -1;
    for (int i=1; i<(int)indegrees.size(); i++) {
        if (indegrees[i] == 0) {
            root_node = i;
        }

    }
    bool no_cycle = dfs(root_node);
    // case 2
    if (no_cycle) {
        vector<int> ans;
        for (auto e: edges) {
            if (e[1] == node_with_2_indegree) {
            //if (e[0] != root_node && e[1] == node_with_2_indegree) {
                ans = e;
            }
        }
        return ans;
    } else {
    // case 3, note that we can not remove edge comming from root
        vector<int> ans;
        for (auto e: edges) {
            if (e[0] != root_node && e[1] == node_with_2_indegree) {
                ans = e;
            }
        }
        return ans;
    }
}


/*
    There are n computers numbered from 0 to n-1 connected by ethernet cables forming a network where connections[i] = [a, b] 
    represents a connection between computers a and b (*undirected*). Any computer can reach any other computer directly or indirectly through the network.

    Given an initial computer network connections. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected. 
    Return the minimum number of times you need to do this in order to make all the computers connected. If it’s not possible, return -1. 

    Hint: use disjoint set to find the number of connected components.
*/
int Solution::makeConnected(int node_count, vector<vector<int>>& connections) {
    // calculate the number of redundant edges which we may use to connect SCC(s) later
    int redundant_edge_cnt = 0;
    DisjointSet dsu(node_count);
    for (auto& p: connections) {
        if (!dsu.unionFunc(p[0], p[1])) {
            redundant_edge_cnt++;
        }
    }
    // calculate the number of SCC(s): `num`
    set<int> groups;
    for (int i=0; i<node_count; ++i) {
        groups.insert(dsu.find(i));
    }
    // we need at least `num-1` edges to connect `num` SCC(s)
    int ans = groups.size()-1;
    return (redundant_edge_cnt>=ans) ? ans : -1;
}


void findRedundantConnection_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<int> actual = ss.findRedundantConnection(graph);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, acutal={}", input, expectedResult, numberVectorToString(actual));
    }
}


void findRedundantDirectedConnection_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<int> actual = ss.findRedundantDirectedConnection(graph);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, acutal={}", input, expectedResult, numberVectorToString(actual));
    }
}


void makeConnected_scaffold(int n, string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.makeConnected(n, graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", n, input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, acutal={}", n, input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running findRedundantConnection tests:");
    TIMER_START(findRedundantConnection);
    findRedundantConnection_scaffold("[[1,2],[2,3],[1,3],[2,4]]", "[1,3]");
    findRedundantConnection_scaffold("[[1,2], [1,3], [2,3]]", "[2,3]");
    findRedundantConnection_scaffold("[[1,2], [2,3], [3,4], [1,4], [1,5]]", "[1,4]");
    TIMER_STOP(findRedundantConnection);
    SPDLOG_WARN("findRedundantConnection tests use {} ms", TIMER_MSEC(findRedundantConnection));

    SPDLOG_WARN("Running findRedundantDirectedConnection tests:");
    TIMER_START(findRedundantDirectedConnection);
    findRedundantDirectedConnection_scaffold("[[1,2],[2,3],[3,4],[4,1],[1,5]]", "[4,1]"); // no duplicate parents, revert to findRedundantConnection
    findRedundantDirectedConnection_scaffold("[[1,2],[1,3],[2,3]]", "[2,3]"); // a node has duplicate parents [u, v], return the latter
    findRedundantDirectedConnection_scaffold("[[1,2],[3,2],[2,3]]", "[3,2]");
    findRedundantDirectedConnection_scaffold("[[3,2],[2,3],[1,2]]", "[3,2]");
    findRedundantDirectedConnection_scaffold("[[3,2],[1,2],[2,3]]", "[3,2]");
    findRedundantDirectedConnection_scaffold("[[1,2],[2,3],[3,4],[1,4],[1,5]]", "[1,4]");
    findRedundantDirectedConnection_scaffold("[[1,2],[2,3],[3,4],[4,2]]", "[4, 2]");
    findRedundantDirectedConnection_scaffold("[[2,1],[3,1],[4,2],[1,4]]", "[2,1]"); // both, return [u1, v] or [u2, v] that creates the loop
    findRedundantDirectedConnection_scaffold("[[3,1],[2,1],[4,2],[1,4]]", "[2,1]");
    findRedundantDirectedConnection_scaffold("[[4,1],[2,3],[3,1],[1,2]]", "[3,1]");
    TIMER_STOP(findRedundantDirectedConnection);
    SPDLOG_WARN("findRedundantDirectedConnection tests use {} ms", TIMER_MSEC(findRedundantDirectedConnection));

    SPDLOG_WARN("Running makeConnected tests:");
    TIMER_START(makeConnected);
    makeConnected_scaffold(5, "[[0,1],[1,2],[2,3]]", -1);
    makeConnected_scaffold(4, "[[0,1],[0,2],[1,2]]", 1);
    makeConnected_scaffold(6, "[[0,1],[0,2],[0,3],[1,2],[1,3]]", 2);
    makeConnected_scaffold(6, "[[0,1],[0,2],[0,3],[1,2]]", -1);
    makeConnected_scaffold(5, "[[0,1],[0,2],[3,4],[2,3]]", 0);
    makeConnected_scaffold(6, "[[0,1],[0,2],[0,3],[1,3],[4,5]]", 1);
    makeConnected_scaffold(100, "[[17,51],[33,83],[53,62],[25,34],[35,90],[29,41],[14,53],[40,84],"
                                "[41,64],[13,68],[44,85],[57,58],[50,74],[20,69],[15,62],[25,88],"
                                "[4,56],[37,39],[30,62],[69,79],[33,85],[24,83],[35,77],[2,73],"
                                "[6,28],[46,98],[11,82],[29,72],[67,71],[12,49],[42,56],[56,65],"
                                "[40,70],[24,64],[29,51],[20,27],[45,88],[58,92],[60,99],[33,46],"
                                "[19,69],[33,89],[54,82],[16,50],[35,73],[19,45],[19,72],[1,79],"
                                "[27,80],[22,41],[52,61],[50,85],[27,45],[4,84],[11,96],[0,99],"
                                "[29,94],[9,19],[66,99],[20,39],[16,85],[12,27],[16,67],[61,80],"
                                "[67,83],[16,17],[24,27],[16,25],[41,79],[51,95],[46,47],[27,51],"
                                "[31,44],[0,69],[61,63],[33,95],[17,88],[70,87],[40,42],[21,42],"
                                "[67,77],[33,65],[3,25],[39,83],[34,40],[15,79],[30,90],[58,95],"
                                "[45,56],[37,48],[24,91],[31,93],[83,90],[17,86],[61,65],[15,48],"
                                "[34,56],[12,26],[39,98],[1,48],[21,76],[72,96],[30,69],[46,80],"
                                "[6,29],[29,81],[22,77],[85,90],[79,83],[6,26],[33,57],[3,65],"
                                "[63,84],[77,94],[26,90],[64,77],[0,3],[27,97],[66,89],[18,77],[27,43]]", 13);
    
    TIMER_STOP(makeConnected);
    SPDLOG_WARN("makeConnected tests use {} ms", TIMER_MSEC(makeConnected));
}
