#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 785, 886, 1042*/
class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph);
    bool possibleBipartition(int N, vector<vector<int>>& dislikes);
    vector<int> gardenNoAdj(int N, vector<vector<int>>& paths);
};


/*
    Given an undirected graph, return true if and only if it is bipartite.
    
    Recall that a graph is bipartite if we can split its set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.
    The graph is given in the following form: graph[i] is a list of indexes (0-index) j for which the edge between nodes i and j exists. 
    There are no self edges or parallel edges: graph[i] does not contain i, and it doesn't contain any element twice.

    The graph may not be connected, meaning there may be two nodes u and v such that there is no path between them.

    as wikipedia puts it: In the mathematical field of graph theory, a bipartite graph (or bigraph) 
    is a graph whose vertices can be divided into two disjoint and independent sets U and V such that 
    every edge connects a vertex in U to one in V. Vertex sets U and V are usually called the parts of the graph. 
    Equivalently, a bipartite graph is a graph that does not contain any odd-length cycles. 
    The two sets U and V may be thought of as a coloring of the graph with two colors: 
    if one colors all nodes in U blue, and all nodes in V green, each edge has endpoints of differing colors, 
    as is required in the graph coloring problem.
*/
bool Solution::isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> visited(n, 0);
    vector<int> color(n, 0); 
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                color[v] = (color[u]==1 ? 2 : 1);
                if (!dfs(v)) {
                    return false;
                }
            } else if (color[u] == color[v]) {
                return false;
            }
        }
        visited[u] = 2;
        return true;
    };
    for (int u=0; u<n; ++u) {
        if (visited[u] == 0) {
            color[u] = 1;
            if (!dfs(u)) {
                return false;
            }
        }
    }
    return true;
}


/*
    Given a set of N people (numbered 1, 2, ..., N), we would like to split everyone into two groups of any size.
    Each person may dislike some other people, and they should not go into the same group. 
    Formally, if dislikes[i] = [a, b], it means it is not allowed to put the people numbered a and b into the same group.
    Return true if and only if it is possible to split everyone into two groups in this way.
*/
bool Solution::possibleBipartition(int N, vector<vector<int>>& dislikes) {

{ // dfs solution
    vector<vector<int>> graph(N);
    for (auto& p: dislikes) {
        graph[p[0]-1].push_back(p[1]-1);
        graph[p[1]-1].push_back(p[0]-1);
    }
    vector<int> visited(N, 0);
    vector<int> colors(N, 0);
    function<bool(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                colors[v] = (colors[u]==1 ? 2 : 1);
                if (!dfs(v)) {
                    return false;
                }
            } else if (colors[u] == colors[v]){
                return false;
            }
        }
        visited[u] = 2;
        return true;
    };
    for (int u=0; u<N; ++u) {
        if (visited[u] == 0) {
            colors[u] = 1;
            if (!dfs(u)) {
                return false;
            }
        }
    }
    return true;
}

{ // bfs solution
    vector<vector<int>> graph(N+1);
    for (auto& p: dislikes) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    queue<int> q; 
    vector<int> colors(N+1, 0); 
    for (int i=1; i<=N; ++i) {
        if (colors[i] != 0) {
            continue;
        }
        q.push(i); colors[i] = 1;
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            for (auto v: graph[u]) {
                if (colors[v] == 0) { // unvisited
                    q.push(v);
                    colors[v] = colors[u]==1 ? 2 : 1; // dye node v with a different color from its parent node u
                } else if (colors[u] == colors[v]) {
                    return false;
                }
            }
        }
    }
    return true;
}

}


/*
    You have N gardens, labelled 1 to N. In each garden, you want to plant one of 4 types of flowers.
    paths[i] = [x, y] means there is a bidirectional path from garden x to garden y. Also, there is no garden that has more than 3 paths coming into or leaving it.
    Your task is to choose a flower type for each garden such that, for any two gardens connected by a path, they have different types of flowers.
    Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)-th garden. The flower types are denoted 1, 2, 3, or 4.
    It is guaranteed an answer exists.
*/
vector<int> Solution::gardenNoAdj(int N, vector<vector<int>>& paths) {

{ // dfs solution
    vector<vector<int>> graph(N);
    for (auto& p: paths) {
        graph[p[0]-1].push_back(p[1]-1);
        graph[p[1]-1].push_back(p[0]-1);
    }
    vector<int> color(N, 0);
    vector<int> visited(N, 0);
    auto choose_color = [&] (int u) {
        vector<bool> mask(4, false);
        for (auto v: graph[u]) {
            if (color[v] != 0) {
                mask[color[v]-1] = true;
            }
        }
        for (int i=0; i<mask.size(); ++i) {
            if (!mask[i]) {
                return i+1;
            }
        }
        return 0;
    };
    function<void(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                color[v] = choose_color(v);
                dfs(v);
            }
        }
        visited[u] = 2;
    };
    for (int u=0; u<N; ++u) {
        if (visited[u] == 0) {
            color[u] = 1;
            dfs(u);
        }
    }
    return color;
}

{ // bfs solution
    vector<vector<int>> graph(N+1);
    for (auto& p: paths) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    vector<int> colors(N, 0);
    for (int u=1; u<=N; ++u) {
        int mask = 0;
        for (auto v: graph[u]) { // colors used by adjacent gardens
            mask |= (1<<colors[v-1]);
        }
        for (int i=1; i<=4; ++i) { // use the first color which is still vacant
            if ((mask & (1<<i))==0) {
                colors[u-1] = i;
                break;
            }
        }
    }
    return colors;
}
}

void isBipartite_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.isBipartite(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual:" << actual;
    }
}

void possibleBipartition_scaffold(int N, string input, bool expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.possibleBipartition(N, graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << N << ", " << input << ", expectedResult<" << expectedResult << ">) passed";
    } else {
        util::Log(logERROR) << "Case(" << N << ", " << input << ", expectedResult<" << expectedResult << ">) failed";
    }
}

void gardenNoAdj_scaffold(int N, string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<int> actual = ss.gardenNoAdj(N, graph);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << N << ", " << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << N << ", " << input << ", " << expectedResult << ") failed, acutal: " << numberVectorToString(actual);
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isBipartite tests: ";
    TIMER_START(isBipartite);
    isBipartite_scaffold("[[1,3],[0,2],[1,3],[0,2]]", true);
    isBipartite_scaffold("[[1,2,3],[0,2],[1,3],[0,2]]", false);
    isBipartite_scaffold("[[1,2,3],[0,2],[0,1,3],[0,2]]", false);
    isBipartite_scaffold("[[],[2,4,6],[1,4,8,9],[7,8],[1,2,8,9],[6,9],[1,5,7,8,9],[3,6,9],[2,3,4,6,9],[2,4,5,6,7,8]]", false);
    isBipartite_scaffold("[[],[3],[],[1],[]]", true);
    TIMER_STOP(isBipartite);
    util::Log(logESSENTIAL) << "isBipartite using " << TIMER_MSEC(isBipartite) << " milliseconds";

    util::Log(logESSENTIAL) << "Running possibleBipartition tests: ";
    TIMER_START(possibleBipartition);
    possibleBipartition_scaffold(4, "[[1,2],[1,3],[2,4]]", true);
    possibleBipartition_scaffold(3, "[[1,2],[1,3],[2,3]]", false);
    possibleBipartition_scaffold(5, "[[1,2],[2,3],[3,4],[4,5],[1,5]]", false);
    possibleBipartition_scaffold(5, "[[1,2],[3,4],[4,5],[3,5]]", false);
    TIMER_STOP(possibleBipartition);
    util::Log(logESSENTIAL) << "possibleBipartition using " << TIMER_MSEC(possibleBipartition) << " milliseconds";

    util::Log(logESSENTIAL) << "Running gardenNoAdj tests: ";
    TIMER_START(gardenNoAdj);
    gardenNoAdj_scaffold(3, "[[1,2],[2,3],[3,1]]", "[1,2,3]");
    gardenNoAdj_scaffold(4, "[[1,2],[3,4]]", "[1,2,1,2]");
    gardenNoAdj_scaffold(4, "[[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]", "[1,2,3,4]");
    TIMER_STOP(gardenNoAdj);
    util::Log(logESSENTIAL) << "gardenNoAdj using " << TIMER_MSEC(gardenNoAdj) << " milliseconds";
}
