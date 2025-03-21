#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 207, 210, 802 */

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites);
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites);
    vector<int> eventualSafeNodes(vector<vector<int>>& graph);
};

/*
    There are a total of n courses you have to take, labelled from 0 to n - 1. Some courses may have prerequisites, 
    for example to take course 0 you have to take course 1 first, which is expressed as a pair: [0, 1]
    Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses? (no cycle dependency found)
    Hint: use dfs(coloring) to detect if there is a cycle in the graph. 
*/
bool Solution::canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    // build a DAG (Directed Acyclic Graph)
    vector<vector<int>> graph(numCourses);
    for (auto& p: prerequisites) {
        graph[p[0]].push_back(p[1]);
    }
    vector<int> visited(numCourses, 0);
    // return false if we find a cycle
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1; // visiting
        for (auto v: graph[u]) {
            if (visited[v] == 0) { // unvisited
                if (!dfs(v)) {
                    return false;
                }
            } else if (visited[v] == 1) { // found a cycle
                return false;
            }
        }
        visited[u] = 2; // visited
        return true;
    };
    // traverse all nodes to see if there is a cycle
    for (int u=0; u<numCourses; ++u) {
        if (visited[u] == 0) {
            if (!dfs(u)) {
                return false;
            }
        }
    }
    return true;
}

/*
    There are a total of n courses you have to take, labelled from 0 to n - 1. Some courses may have prerequisites,
    for example to take course 0 you have to take course 1 first, which is expressed as a pair: [0, 1]
    Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.
    There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.
*/
vector<int> Solution::findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    // build a DAG (Directed Acyclic Graph)
    vector<vector<int>> graph(numCourses);
    for (auto& p: prerequisites) {
        graph[p[0]].push_back(p[1]);
    }
    vector<int> courses;
    vector<int> visited(numCourses, 0);
    // return false if we find a cycle
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1; // visiting
        for (auto v: graph[u]) {
            if (visited[v] == 0) { // unvisited
                if (!dfs(v)) {
                    return false;
                }
            } else if (visited[v] == 1) { // found a cycle
                return false;
            }
        }
        // start lesson u when all its prerequisites are finished
        courses.push_back(u);
        visited[u] = 2; // visited
        return true;
    };
    // traverse all nodes to see if there is a cycle
    for (int u=0; u<numCourses; ++u) {
        if (visited[u] == 0) {
            if (!dfs(u)) {
                courses.clear();
                return courses;
            }
        }
    }
    return courses;
}


/*
    In a directed graph, we start at some node and walk along a directed edge of the graph at every turn.
    If we reach a node that is terminal (that is, it has no outgoing directed edges), we stop.
    Now, say our starting node is eventually safe if and only if we must eventually walk to a terminal node. More specifically,
    there exists a natural number K so that for any choice of where to walk, we must have stopped at a terminal node in less than K steps.

    Which nodes are eventually safe? Return them as an array in sorted order.

    The directed graph has N nodes with labels 0, 1, ..., N-1, where N is the length of graph.
    The graph is given in the following form: graph[i] is a list of labels j such that (i, j) is a directed edge of the graph.

    Hint: remove nodes in a connected component which contains cycle(s) but note that a terminal node is always safe itself.
*/
vector<int> Solution::eventualSafeNodes(vector<vector<int>>& graph) {

// naive way
{    
    int n = graph.size();
    vector<int> visited(n, 0);
    // return true if node u is eventually safe
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1; // visiting
        for (auto v: graph[u]) {
            if (visited[v] == 0) { // unvisited
                if (!dfs(v)) {
                    return false;
                }
            } else if (visited[v] == 1) { // found a cycle
                return false;
            }
        }
        visited[u] = 2; // visited
        return true;
    };
    for (int u=0; u<n; ++u) {
        if (visited[u] == 0) {
            dfs(u);
        }
    }
    vector<int> ans;
    for (int u=0; u<n; ++u) {
        if (visited[u] == 2) {
            ans.push_back(u);
        }
    }
    return ans;
}

// naive way
{    
    vector<int> ans;
    int n = graph.size();
    vector<int> visited(n, 0);
    // return true if node u is eventually safe
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1; // visiting
        for (auto v: graph[u]) {
            if (visited[v] == 0) { // unvisited
                if (!dfs(v)) {
                    return false;
                }
            } else if (visited[v] == 1) { // found a cycle
                return false;
            }
        }
        ans.push_back(u);
        visited[u] = 2; // visited
        return true;
    };
    for (int u=0; u<n; ++u) {
        if (visited[u] == 0) {
            dfs(u);
        }
    }
    std::sort(ans.begin(), ans.end());
    return ans;
}

}


void canFinish_scaffold(int numCourses, string input, bool expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.canFinish(numCourses, graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", numCourses, input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", numCourses, input, expectedResult, actual);
    }
}


void findOrder_scaffold(int numCourses, string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<int> actual = ss.findOrder(numCourses, graph);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", numCourses, input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual={}", numCourses, input, expectedResult, numberVectorToString<int>(actual));
    }
}


void eventualSafeNodes_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<int> actual = ss.eventualSafeNodes(graph);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, numberVectorToString<int>(actual));
    }
}


int main() {
    SPDLOG_WARN("Running canFinish tests:");
    TIMER_START(canFinish);
    canFinish_scaffold(2, "[[1,0]]", true);
    canFinish_scaffold(2, "[[0,0]]", false);
    canFinish_scaffold(2, "[[1,0],[0,1]]", false);
    canFinish_scaffold(8, "[[1,0],[2,6],[1,7],[5,1],[6,4],[7,0],[0,5]]", false);
    canFinish_scaffold(8, "[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]]", true);
    TIMER_STOP(canFinish);
    SPDLOG_WARN("canFinish tests use {} ms", TIMER_MSEC(canFinish));

    SPDLOG_WARN("Running findOrder tests:");
    TIMER_START(findOrder);
    findOrder_scaffold(2, "[[1,0]]", "[0,1]");
    findOrder_scaffold(2, "[[0,0]]", "[]");
    findOrder_scaffold(2, "[[1,0],[0,1]]", "[]");
    findOrder_scaffold(8, "[[1,0],[2,6],[1,7],[5,1],[6,4],[7,0],[0,5]]", "[]");
    findOrder_scaffold(8, "[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]]", "[5,0,7,1,4,6,2,3]");
    TIMER_STOP(findOrder);
    SPDLOG_WARN("findOrder tests use {} ms", TIMER_MSEC(findOrder));

    SPDLOG_WARN("Running eventualSafeNodes tests:");
    TIMER_START(eventualSafeNodes);
    eventualSafeNodes_scaffold("[[1,2],[2,3],[5],[0],[5],[],[]]", "[2,4,5,6]");
    eventualSafeNodes_scaffold("[[1],[2,4],[3],[2],[]]", "[4]");
    eventualSafeNodes_scaffold("[[1,2,3,4],[1,2],[3,4],[0,4],[]]", "[4]");
    TIMER_STOP(eventualSafeNodes);
    SPDLOG_WARN("eventualSafeNodes tests use {} ms", TIMER_MSEC(eventualSafeNodes));
}
