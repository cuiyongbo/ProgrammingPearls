#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 433, 815, 863, 1129,  1263*/

class Solution {
public:
    int minMutation(string start, string end, vector<string>& bank);
    int numBusesToDestination(vector<vector<int>>& routes, int S, int T);
    vector<int> distanceK(TreeNode* root, int target, int K);
    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges);
    int minPushBox(vector<vector<char>>& grid);

private:
    int numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T);
    int numBusesToDestination_dsu(vector<vector<int>>& routes, int S, int T);
    int numBusesToDestination_napolen(vector<vector<int>>& routes, int S, int T);
};

int Solution::minMutation(string start, string end, vector<string>& bank) {
    /*
        A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".

        Suppose we need to investigate about a mutation (mutation from “start” to “end”), where ONE mutation 
        is defined as ONE single character changed in the gene string.

        For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.

        Also, there is a given gene “bank”, which records all the valid gene mutations.
        A gene must be in the bank to make it a valid gene string.

        Now, given 3 things – start, end, bank, your task is to determine what is the minimum number 
        of mutations needed to mutate from “start” to “end”. If there is no such a mutation, return -1.

        Note:

            Starting point is assumed to be valid, so it might not be included in the bank.
            If multiple mutations are needed, all mutations during in the sequence must be valid.
            You may assume start and end string is not the same.
    */

    auto isValidMutation = [](const string& s, const string& b) {
        int count = 0;
        for (int i=0; i<(int)s.length(); ++i) {
            if (s[i] != b[i]) {
                ++count;
            }
        }
        return count == 1;
    };

    unordered_set<string> visited;
    visited.emplace(start);

    int steps = 0;
    queue<string> q; q.push(start);
    while (!q.empty()) {
        int size = (int)q.size();
        for (int i=0; i<size; ++i) {
            auto s = q.front(); q.pop();
            if (s == end) {
                return steps;
            }
            for(const auto& b: bank) {
                if (visited.count(b) == 0 && isValidMutation(s, b)) {
                    visited.emplace(b);
                    q.push(b);
                } 
            }
        }
        ++steps;
    }
    return -1;
}

int Solution::numBusesToDestination(vector<vector<int>>& routes, int S, int T) {
    /*
        We have a list of bus routes. Each routes[i] is a bus route that the i-th bus repeats forever. 
        For example if routes[0] = [1, 5, 7], this means that the first bus (0-th indexed) travels 
        in the sequence 1->5->7->1->5->7->1->… forever.

        We start at bus stop S (initially not on a bus), and we want to go to bus stop T. 
        Travelling by buses only, what is the least number of buses we must take to reach 
        our destination? Return -1 if it is not possible.
    */

    return numBusesToDestination_bfs(routes, S, T);
    //return numBusesToDestination_dsu(routes, S, T);
    //return numBusesToDestination_napolen(routes, S, T);
}

int Solution::numBusesToDestination_napolen(vector<vector<int>>& routes, int S, int T)
{
    // build an undirected graph
    // node: u, bus labelled u
    // edge: (u,v), bus[u] and bus[v] have at least one stop in common
    // S in buses(u1, u2, ...), T in buses(v1, v2, ...)
    // find the pair(u, v) which has the shortest length
    // run floyd-warshall algorithm

    int busCount = (int)routes.size();
    map<int, vector<int>> stationToBusMap;
    for(int i=0; i<busCount; ++i)
    {
        for(const auto& r: routes[i])
        {
            stationToBusMap[r].push_back(i);
        }
    }

    vector<vector<int>> distance_table(busCount, vector<int>(busCount, INT32_MAX));

    // initialize distance table as weight function
    for(const auto& it: stationToBusMap)
    {
        size_t count = it.second.size();
        for(size_t i=0; i<count; ++i)
        {
            for(size_t j=i+1; j<count; ++j)
            {
                distance_table[it.second[i]][it.second[j]] = 1;
                distance_table[it.second[j]][it.second[i]] = 1;
            }
        }
    }

    for(int k=0; k<busCount; ++k)
    {
        distance_table[k][k] = 0;
        for (int i = 0; i < busCount; ++i)
        {
            for (int j = 0; j < busCount; ++j)
            {
                if(distance_table[i][k] != INT32_MAX && distance_table[k][j] != INT32_MAX)
                    distance_table[i][j] = std::min(distance_table[i][j], distance_table[i][k] + distance_table[k][j]);
            }
        }
    }
    
    int ans = INT32_MAX;
    for(const auto& s: stationToBusMap[S])
    {
        for(const auto& t: stationToBusMap[T])
        {
            ans = std::min(ans, distance_table[s][t]);
        }
    }

    return ans == INT32_MAX ? -1 : ans+1;
}

int Solution::numBusesToDestination_dsu(vector<vector<int>>& routes, int S, int T)
{
    int busCount = (int)routes.size();
    map<int, int> stationToBusGroup;
    map<int, vector<int>> stationToBusMap;
    DisjointSet dsu(busCount);
    for(int i=0; i<busCount; ++i)
    {
        for(const auto& n: routes[i])
        {
            if(stationToBusGroup[n] != 0)
            {
                dsu.unionFunc(stationToBusGroup[n], i+1);
            }
            stationToBusGroup[n] = i+1;
            stationToBusMap[n].push_back(i);
        }
    }

    // is S and T in the same connected component?
    int group = dsu.find(stationToBusGroup[S]);
    if(group != dsu.find(stationToBusGroup[T]))
        return -1;

    // exclude disjoint components
    vector<bool> visited(busCount, false);
    for(int i=0; i<busCount; ++i)
    {
        if(group != dsu.find(i+1))
            visited[i] = true;
    }

    int steps = 0;
    queue<int> q;
    q.push(S);
    while(!q.empty())
    {
        size_t size = q.size();
        for(size_t i=0; i != size; ++i)
        {
            auto u = q.front(); q.pop();
            if(u == T) return steps;
            for(auto v: stationToBusMap[u])
            {
                if(visited[v]) continue;
                visited[v] = true;
                for(auto r: routes[v])
                    q.push(r);
            }
        }
        ++steps;
    }

    return -1;
}

int Solution::numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T) {
    map<int, vector<int>> stationToBusMap;
    for (int i=0; i<(int)routes.size(); ++i) {
        for (const auto& n: routes[i]) {
            stationToBusMap[n].push_back(i);
        }
    }
    int steps = 0;
    queue<int> q; q.push(S);
    vector<bool> visited(routes.size(), false);
    while (!q.empty()) {
        int size = (int)q.size();
        for (int i=0; i<size; ++i) {
            auto u = q.front(); q.pop();
            if(u == T) {
                return steps;
            } 
            for (const auto& r: stationToBusMap[u]) {
                if (!visited[r]) {
                    visited[r] = true;
                    for (const auto& v: routes[r]) {
                        q.push(v);
                    }
                }
            }
        }
        ++steps;
    }
    return -1;
}

vector<int> Solution::distanceK(TreeNode* root, int target, int K) {
    /*
        We are given a binary tree (with root node root), a target node, and an integer value K.
        Return a list of the values of all nodes that have a distance K from the target node. 
        The answer can be returned in any order.

        Hint: convert the tree into a undirected graph, and perform bfs search from target, 
        return the nodes at the Kth traversal.
    */

    map<int, vector<int>> graph;
    function<void(TreeNode*)> build_graph = [&] (TreeNode* node) {
        if (node != nullptr) {
            if (node->left != nullptr) {
                graph[node->val].push_back(node->left->val);
                graph[node->left->val].push_back(node->val);
                build_graph(node->left);
            }
            if (node->right != nullptr) {
                graph[node->val].push_back(node->right->val);
                graph[node->right->val].push_back(node->val);
                build_graph(node->right);
            }
        }
    };

    build_graph(root);

    set<int> visited;
    visited.insert(target);

    int steps = 0;
    vector<int> ans;
    queue<int> q; q.push(target);
    while (!q.empty()) {
        ans.clear();
        size_t size = q.size();
        for (size_t i=0; i != size; ++i) {
            auto u = q.front(); q.pop();
            for (const auto& v: graph[u]) {
                if (visited.count(v) == 0) {
                    visited.insert(v);
                    ans.push_back(v);
                    q.push(v);
                }
            }
        }
        if(++steps == K) {
            break;
        }
    }

    // sort is not necessary, but performing a sort make it much easier for test
    std::sort(ans.begin(), ans.end());
    return ans;
}

vector<int> Solution::shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges) {
    /*
        Consider a directed graph, with nodes labelled 0, 1, ..., n-1.
        In this graph, each edge is either red or blue, and there could be self-edges or parallel edges.
        Each [i, j] in red_edges denotes a red directed edge from node i to node j.  
        Similarly, each [i, j] in blue_edges denotes a blue directed edge from node i to node j.
        Return an array answer of length n, where each answer[X] is the length of the shortest path 
        from node 0 to node X such that the edge colors alternate along the path (or -1 if such a path doesn’t exist).
    */

    vector<vector<int>> graphs[2];

    // red graph
    graphs[0].resize(n);
    for (const auto& e: red_edges) {
        graphs[0][e[0]].push_back(e[1]);
    }

    // blue graph
    graphs[1].resize(n); 
    for (const auto& e: blue_edges) {
        graphs[1][e[0]].push_back(e[1]);
    }

    vector<int> ans(n, INT32_MAX);
    ans[0] = 0;

    auto search = [&](int color) {
        if (graphs[color][0].empty()) {
            return;
        }
        
        int steps = 0;
        set<int> visited_set[2];
        visited_set[color].insert(0);
        queue<int> q; q.push(0);
        while (!q.empty()) {
            size_t size = q.size();
            while (size-- > 0) {
                auto u = q.front(); q.pop();
                ans[u] = std::min(ans[u], steps);
                for (const auto& v: graphs[color][u]) {
                    if (visited_set[color].count(v) == 0) {
                        visited_set[color].insert(v);
                        q.push(v);
                    }
                }
            }
            color = !color; // reverse color
            ++steps;
        }
    };

    // search from red color
    search(0);
    // search from blue color
    search(1);
    std::transform(ans.begin(), ans.end(), ans.begin(), [](int n) {return n==INT32_MAX ? -1 : n;});
    return ans;
}

int Solution::minPushBox(vector<vector<char>>& grid) {
    /*
        Storekeeper is a game in which the player pushes boxes around in a warehouse trying to get them to target locations.

        The game is represented by a grid of size n*m, where each element is a wall, floor, or a box.

        Your task is move the box 'B' to the target position 'T' under the following rules:

            Player is represented by character 'S' and can move up, down, left, right in the grid if it is a floor (empy cell).
            Floor is represented by character '.' that means a free cell to walk.
            Wall is represented by character '#' that means an obstacle  (impossible to walk there). 
            There is only one box 'B' and one target cell 'T' in the grid.
            The box can be moved to an adjacent free cell by standing next to the box and then moving in the direction of the box. This is a push.
            The player cannot walk through the box.

        Return the minimum number of pushes to move the box to the target. If there is no way to reach the target, return -1.
    */

    int rows = grid.size();
    int columns = grid[0].size();

    int id = 0;
    int box=0, player=0, dest=0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 'S') {
                player = id;
            }
            if (grid[r][c] == 'T') {
                dest = id;
            }
            if (grid[r][c] == 'B') {
                box = id;
            }
            ++id;
        }
    }

    const vector<vector<int>> MOVES {{-columns, columns}, {columns, -columns}, {-1, 1}, {1, -1}};
    auto hasPath = [&](int box, int player, const vector<int>& move) {
        queue<int> q; q.push(player);
        unordered_set<int> visited; visited.emplace(player);
        while (!q.empty()) {   
            int size = (int)q.size();
            for (int i=0; i<size; ++i) {
                auto t = q.front(); q.pop();
                if(t == box + move[1]) {
                    // ok, play got to a position to push box
                    return true;
                }
                for (const auto& m: MOVES) {
                    int s = t + m[0];
                    if (s<0 || s>=rows*columns) {
                        continue;
                    }
                    int r = s / columns;
                    int c = s % columns;
                    if (grid[r][c] == '#' || s == box || visited.count(s) != 0) {
                        continue;
                    }
                    visited.emplace(s);
                    q.push(s);
                }
            }
        }
        return false;
    };

    int steps = 0;
    queue<int> q; q.push(box);
    unordered_set<int> visited; visited.emplace(box);
    while (!q.empty()) {
        int size = (int)q.size();
        for (int i=0; i<size; ++i) {
            auto t = q.front(); q.pop();
            if (t == dest) {
                return steps;
            }
            for (const auto& m: MOVES) {
                int s = t + m[0];
                if (s<0 || s>=rows*columns) {
                    continue;
                }
                int r = s / columns;
                int c = s % columns;
                if (grid[r][c] == '#' || visited.count(s) != 0) {
                    continue;
                }
                // can player push t?
                if (hasPath(t, player, m)) {
                    visited.emplace(s);
                    q.push(s);
                }
            }
        }
        ++steps;
    }
    return -1;
}

void minMutation_scaffold(string input1, string input2, string input3, int expectedResult) {
    Solution ss;
    vector<string> bank = stringTo1DArray<string>(input3);
    int actual = ss.minMutation(input1, input2, bank);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << actual;
    }
}

void numBusesToDestination_scaffold(string input1, int input2, int input3, int expectedResult) {
    Solution ss;
    vector<vector<int>> routes = stringTo2DArray<int>(input1);
    int actual = ss.numBusesToDestination(routes, input2, input3);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << actual;
    }
}

void distanceK_scaffold(string input1, int input2, int input3, string expectedResult) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input1);
    vector<int> actual = ss.distanceK(root, input2, input3);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << numberVectorToString(actual);
    }
}

void shortestAlternatingPaths_scaffold(int input1, string input2, string input3, string expectedResult) {
    Solution ss;
    vector<vector<int>> red_edges = stringTo2DArray<int>(input2);
    vector<vector<int>> blue_edges = stringTo2DArray<int>(input3);
    vector<int> actual = ss.shortestAlternatingPaths(input1, red_edges, blue_edges);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << numberVectorToString(actual);
    }
}

void minPushBox_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<char>> grid = stringTo2DArray<char>(input);
    int actual = ss.minPushBox(grid);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running minMutation tests:";
    TIMER_START(minMutation);
    minMutation_scaffold("AACCGGTT", "AACCGGTA", "[AACCGGTA]", 1);
    minMutation_scaffold("AACCGGTT", "AAACGGTA", "[AACCGGTA, AACCGCTA, AAACGGTA]", 2);
    minMutation_scaffold("AAAAACCC", "AACCCCCC", "[AAAACCCC, AAACCCCC, AACCCCCC]", 3);
    TIMER_STOP(minMutation);
    util::Log(logESSENTIAL) << "minMutation using " << TIMER_MSEC(minMutation) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numBusesToDestination tests:";
    TIMER_START(numBusesToDestination);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 5, -1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 3, 1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 2, 5, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 1, 6, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 2, 3, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 6, 3, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 7, 1, 1);
    TIMER_STOP(numBusesToDestination);
    util::Log(logESSENTIAL) << "numBusesToDestination using " << TIMER_MSEC(numBusesToDestination) << " milliseconds";

    util::Log(logESSENTIAL) << "Running distanceK tests:";
    TIMER_START(distanceK);
    distanceK_scaffold("[3,5,1,6,2,0,8,null,null,7,4]", 5, 2, "[1,4,7]");
    TIMER_STOP(distanceK);
    util::Log(logESSENTIAL) << "distanceK using " << TIMER_MSEC(distanceK) << " milliseconds";

    util::Log(logESSENTIAL) << "Running shortestAlternatingPaths tests:";
    TIMER_START(shortestAlternatingPaths);
    shortestAlternatingPaths_scaffold(3, "[[0,1],[1,2]]", "[]", "[0,1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[0,1]]", "[[2,1]]", "[0,1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[1,0]]", "[[2,1]]", "[0,-1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[0,1]]", "[[1,2]]", "[0,1,2]");
    shortestAlternatingPaths_scaffold(3, "[[0,1],[0,2]]", "[[1,0]]", "[0,1,1]");
    TIMER_STOP(shortestAlternatingPaths);
    util::Log(logESSENTIAL) << "shortestAlternatingPaths using " << TIMER_MSEC(shortestAlternatingPaths) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minPushBox tests:";
    TIMER_START(minPushBox);

    string grid = R"([[#,#,#,#,#,#],
                   [#,T,#,#,#,#],
                   [#,.,.,B,.,#],
                   [#,.,#,#,.,#],
                   [#,.,.,.,S,#],
                   [#,#,#,#,#,#]])";
    minPushBox_scaffold(grid, 3);

    grid = R"([[#,#,#,#,#,#,#],
               [#,S,#,.,B,T,#],
               [#,#,#,#,#,#,#]])";
    minPushBox_scaffold(grid, -1);

    grid = R"([[#,#,#,#,#,#],
               [#,T,.,.,#,#],
               [#,.,#,B,.,#],
               [#,.,.,.,.,#],
               [#,.,.,.,S,#],
               [#,#,#,#,#,#]])";
    minPushBox_scaffold(grid, 5);

    grid = R"([[#,#,#,#,#,#],
               [#,T,#,#,#,#],
               [#,.,.,B,.,#],
               [#,#,#,#,.,#],
               [#,.,.,.,S,#],
               [#,#,#,#,#,#]])";
    minPushBox_scaffold(grid, -1);

    TIMER_STOP(minPushBox);
    util::Log(logESSENTIAL) << "minPushBox using " << TIMER_MSEC(minPushBox) << " milliseconds";

}
