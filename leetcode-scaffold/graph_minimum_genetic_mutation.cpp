#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 433, 815, 863, 1129, 1263*/

class Solution {
public:
    int minMutation(string start, string end, vector<string>& bank);
    int numBusesToDestination(vector<vector<int>>& routes, int S, int T);
    vector<int> distanceK(TreeNode* root, int target, int K);
    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges);
    int minPushBox(vector<vector<char>>& grid);

private:
    int numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T);
    int numBusesToDestination_napolen(vector<vector<int>>& routes, int S, int T);
};


/*
    A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".
    Suppose we need to investigate about a mutation (mutation from “start” to “end”), where ONE mutation 
    is defined as ONE single character changed in the gene string. For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
    Also, there is a given gene “bank”, which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string.
    Now, given start, end, bank, your task is to determine what is the minimum number of mutations needed to mutate from “start” to “end”. 
    If there is no such a mutation, return -1.
    Note:
        Starting point is assumed to be valid, so it might not be included in the bank.
        If multiple mutations are needed, all mutations during in the sequence must be valid.
        You may assume start and end string is not the same.
*/
int Solution::minMutation(string start, string end, vector<string>& bank) {
    auto is_valid = [&] (string u, string v) {
        int diff = 0;
        for (int i=0; i<u.size() && diff<2; ++i) {
            if (u[i] != v[i]) {
                ++diff;
            }
        }
        return diff == 1;
    };
    // use bfs to find the minimum steps from start to end
    int steps = 0;
    queue<string> q; q.push(start);
    set<string> visited; visited.insert(start);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            if (u == end) { // we reach the end
                return steps;
            }
            for (auto& v: bank) {
                if (visited.count(v) != 0) {
                    continue;
                }
                if (is_valid(u, v)) {
                    q.push(v);
                    visited.insert(v);
                }
            }
        }
        ++steps;
    }
    return -1;
}


/*
    We have a list of bus routes. Each routes[i] is a bus route that the i-th bus repeats forever. 
    For example if routes[0] = [1, 5, 7], this means that the first bus (0-th indexed) travels in the sequence 1->5->7->1->5->7->1->… forever.
    We start at bus stop S (initially not on a bus), and we want to go to bus stop T. 
    Travelling by buses only, what is the least number of buses we must take to reach our destination? Return -1 if it is not possible. [最少换乘]
*/
int Solution::numBusesToDestination(vector<vector<int>>& routes, int S, int T) {
    return numBusesToDestination_bfs(routes, S, T);
    //return numBusesToDestination_napolen(routes, S, T);
}


int Solution::numBusesToDestination_napolen(vector<vector<int>>& routes, int S, int T) {
    // build an undirected graph
    // node: u, bus labelled u
    // edge: (u,v), bus[u] and bus[v] have at least one stop in common
    // S in buses(u1, u2, ...), T in buses(v1, v2, ...)
    // find the pair(u, v) which has the shortest length
    // run floyd-warshall algorithm

    int bus_count = routes.size();
    map<int, vector<int>> station_to_bus;
    for (int i=0; i<bus_count; ++i) {
        for (auto& r: routes[i]) {
            station_to_bus[r].push_back(i);
        }
    }

    // initialize distance table as weight function
    vector<vector<int>> distance_table(bus_count, vector<int>(bus_count, INT32_MAX));
    for (auto& it: station_to_bus) {
        size_t count = it.second.size();
        for (size_t i=0; i<count; ++i) {
            for (size_t j=i+1; j<count; ++j) {
                distance_table[it.second[i]][it.second[j]] = 1;
                distance_table[it.second[j]][it.second[i]] = 1;
            }
        }
    }

    for (int k=0; k<bus_count; ++k) {
        distance_table[k][k] = 0;
        for (int i = 0; i < bus_count; ++i) {
            for (int j = 0; j < bus_count; ++j) {
                if (distance_table[i][k] != INT32_MAX && distance_table[k][j] != INT32_MAX)
                    distance_table[i][j] = std::min(distance_table[i][j], distance_table[i][k] + distance_table[k][j]);
            }
        }
    }
    
    int ans = INT32_MAX;
    for (auto s: station_to_bus[S]) {
        for (auto t: station_to_bus[T]) {
            ans = std::min(ans, distance_table[s][t]);
        }
    }

    return ans == INT32_MAX ? -1 : ans+1;
}


int Solution::numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T) {

{ // naive bfs
    int buses = routes.size();
    // build bus station map
    map<int, vector<int>> bus_station_map;
    for (int i=0; i<buses; i++) {
        for (auto r: routes[i]) {
            bus_station_map[r].push_back(i);
        }
    }
    auto is_intersected = [&] (int i, int j) {
        for (auto r: routes[i]) {
            for (auto l: routes[j]) {
                if (r == l) {
                    return true;
                }
            }
        }
        return false;
    };
    // build a bus graph
    map<int, vector<int>> graph;
    for (int i=0; i<buses; i++) {
        for (int j=i+1; j<buses; j++) {
            if (is_intersected(i, j)) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
    // use bfs to find the minimum bus to travel from S to T
    int steps = 0;
    queue<int> q;
    for (auto r: bus_station_map[S]) {
        q.push(r);
    }
    set<int> visited(bus_station_map[S].begin(), bus_station_map[S].end());
    set<int> ends(bus_station_map[T].begin(), bus_station_map[T].end());
    while (!q.empty()) {
        for (int k=q.size(); k!=0; k--) {
            auto u = q.front(); q.pop();
            if (ends.count(u)) {
                return steps+1;
            }
            for (auto v: graph[u]) {
                if (visited.count(v) == 0) {
                    q.push(v);
                    visited.insert(v);
                }
            }  
        }
        steps++;
    }
    return -1;
}

{ // naive bfs
    map<int, vector<int>> station_to_bus;
    for (int i=0; i<routes.size(); ++i) {
        for (auto s: routes[i]) {
            station_to_bus[s].push_back(i);
        }
    }
    int steps = 0;
    set<int> visited; visited.insert(S);
    queue<int> q; q.push(S);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            if (u == T) {
                return steps;
            }
            for (auto bus: station_to_bus[u]) {
                for (auto stat: routes[bus]) {
                    if (visited.count(stat) == 0) {
                        q.push(stat);
                        visited.insert(stat);
                    }
                }
            }
        }
        ++steps;
    }
    return -1;
}

{ // refined bfs
    int sz = routes.size();
    map<int, vector<int>> station_to_bus;
    for (int i=0; i<sz; ++i) {
        for (auto n: routes[i]) {
            station_to_bus[n].push_back(i);
        }
    }
    int steps = 0;
    queue<int> q; q.push(S);
    vector<bool> visited(sz, false);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            if (u == T) {
                return steps;
            }
            for (auto bus: station_to_bus[u]) {
                if (!visited[bus]) {
                    visited[bus] = true;
                    for (auto v: routes[bus]) {
                        q.push(v);
                    }
                }
            }
        }
        ++steps;
    }
}

}


/*
    We are given a binary tree (with root node root), a target node, and an integer value K.
    Return a list of the values of all nodes that have a distance K from the target node. 
    The answer can be returned in any order.
    Constraints:
        All the values Node.val are unique.
        target is the value of one of the nodes in the tree.
    Hint: convert the tree into a undirected graph, and perform bfs search from target, return the nodes at the Kth traversal.
*/
vector<int> Solution::distanceK(TreeNode* root, int target, int distance) {
    map<int, vector<int>> graph;
    // perform post-order traversal to build a undirected graph
    function<void(TreeNode*)> tree_to_graph = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        tree_to_graph(node->left);
        tree_to_graph(node->right);
        if (node->left != nullptr) {
            graph[node->val].push_back(node->left->val);
            graph[node->left->val].push_back(node->val);
        }
        if (node->right != nullptr) {
            graph[node->val].push_back(node->right->val);
            graph[node->right->val].push_back(node->val);
        }
    };
    tree_to_graph(root);
    // perform bfs to find the destination nodes
    int steps = 0;
    queue<int> q; q.push(target);
    set<int> visited; visited.insert(target);
    while (!q.empty() && steps!=distance) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            for (auto v: graph[u]) {
                if (visited.count(v) == 0) {
                    q.push(v);
                    visited.insert(v);
                }
            }
        }
        ++steps;
    }
    vector<int> ans;
    while (!q.empty()) {
        ans.push_back(q.front()); q.pop();
    }
    return ans;
}


/*
    Consider a *directed* graph, with nodes labelled 0, 1, ..., n-1.
    In this graph, each edge is either red or blue, and there could be self-edges or parallel edges.
    Each [i, j] in red_edges denotes a red directed edge from node i to node j.  
    Similarly, each [i, j] in blue_edges denotes a blue directed edge from node i to node j.
    Return an array answer of length n, where each answer[X] is the length of the shortest path 
    from node 0 to node X such that the edge colors alternate along the path (or -1 if such a path doesn’t exist).
*/
vector<int> Solution::shortestAlternatingPaths(int n, vector<vector<int>>& red_edges, vector<vector<int>>& blue_edges) {
    const int RED = 0;
    const int BLUE = 1;
    // build graph
    vector<vector<int>> graphs[2];
    // red graph
    graphs[RED].resize(n);
    for (auto& e: red_edges) {
        graphs[RED][e[0]].push_back(e[1]);
    }
    // blue graph
    graphs[BLUE].resize(n);
    for (auto& e: blue_edges) {
        graphs[BLUE][e[0]].push_back(e[1]);
    }
    vector<int> ans(n, INT32_MAX);
    auto search = [&] (int color) {
        int steps = 0;
        vector<vector<bool>> visited(2, vector<bool>(n, false));
        visited[color][0] = true;
        queue<int> q; q.push(0);
        while (!q.empty()) {
            for (int k=q.size(); k!=0; --k) {
                auto u = q.front(); q.pop();
                ans[u] = std::min(ans[u], steps);
                //printf("graphs[%d][%d]:%d\n", color, u, graphs[color][u].size());
                for (auto v: graphs[color][u]) {
                    if (!visited[color][v]) {
                        q.push(v);
                        visited[color][v] = true;
                    }
                }
            }
            ++steps;
            color = color==RED ? BLUE : RED; // change color at each step
        }
    };
    search(RED); // search starting from red graph
    search(BLUE); // search starting from blue graph
    std::transform(ans.begin(), ans.end(), ans.begin(), [](int v) {return v==INT32_MAX ? -1 : v;});
    return ans;
}


/*
    Storekeeper is a game in which the player pushes boxes around in a warehouse trying to get them to target locations.
    The game is represented by a grid of size n*m, where each element is a wall, floor, or a box.
    Your task is move the box 'B' to the target position 'T' under the following rules:
        Player is represented by character 'S' and can move up, down, left, right in the grid if it is a floor (empy cell).
        Floor is represented by character '.' that means a free cell to walk.
        Wall is represented by character '#' that means an obstacle (impossible to walk through). 
        The box can be moved to an adjacent free cell by standing next to the box and then moving in the direction of the box. This is a push.
        The player cannot walk through the box.
    Return the minimum number of pushes to move the box to the target. If there is no way to reach the target, return -1.
    Constraints:
        grid contains only characters '.', '#', 'S', 'T', or 'B'.
        There is only one character 'S', 'B', and 'T' in the grid.
*/
int Solution::minPushBox(vector<vector<char>>& grid) {
    int rows = grid.size();
    int columns = grid[0].size();
    using Coordinate = std::pair<int, int>;
    // find player, dest, and box positions
    Coordinate box, player, dest;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 'S') {
                player = std::make_pair(r, c);
            } else if (grid[r][c] == 'T') {
                dest = std::make_pair(r, c);
            } else if (grid[r][c] == 'B') {
                box = std::make_pair(r, c);
            }
        }
    }
    // return true if `push_pos` is reachable from `player_pos`, otherwise false
    auto is_valid_move = [&] (Coordinate player_pos, Coordinate push_pos, Coordinate box_pos) {
        queue<Coordinate> q; q.push(player_pos);
        set<Coordinate> visited; visited.insert(player_pos);
        while (!q.empty()) {
            for (int k=q.size(); k!=0; --k) {
                auto u = q.front(); q.pop();
                if (u == push_pos) {
                    return true;
                }
                for (auto d: directions) {
                    int nr = u.first + d.first;
                    int nc = u.second + d.second;
                    if (nr<0 || nr>=rows || nc<0 || nc>=columns || grid[nr][nc]=='#') {
                        continue;
                    }
                    auto v = make_pair(nr, nc);
                    if (v == box_pos // cannot walk through box, we may compact code by adding box_pos to visited
                        || visited.count(v) != 0) {
                        continue;
                    }
                    q.push(v);
                    visited.insert(v);
                }                
            }
        }
        return false;
    };
    // perform bfs search
    int steps = 0;
    set<pair<Coordinate, Coordinate>> visited; // box_pos, direction from which box is pushed to box_pos
    queue<pair<Coordinate, Coordinate>> q; q.emplace(box, player); // box_pos, player_pos
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto candidate = q.front(); q.pop();
            auto player_pos = candidate.second;
            auto box_pos = candidate.first;
            if (box_pos == dest) {
                return steps;
            }
            for (auto d: directions) {
                int nr = box_pos.first + d.first;
                int nc = box_pos.second + d.second;
                if (nr<0 || nr>=rows || nc<0 || nc>=columns || grid[nr][nc]=='#') {
                    continue;
                }
                auto new_box = make_pair(nr, nc);
                auto pp = make_pair(new_box, d);
                if (visited.count(pp) != 0) {
                    continue;
                }
                // to push box to `new_box`, player must goes to `push_pos` then push
                auto push_pos = make_pair(box_pos.first-d.first, box_pos.second-d.second);
                if (push_pos.first<0 || push_pos.first>=rows || push_pos.second<0 || push_pos.second>=columns || grid[push_pos.first][push_pos.second]=='#') {
                    continue;
                }
                if (is_valid_move(player_pos, push_pos, box_pos)) {
                    visited.insert(pp); // DON'T move it out of `is_valid_move`
                    q.emplace(new_box, box_pos); // player takes the place of previous box position after pushing
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
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input1, input2, input3, expectedResult, actual);
    }
}


void numBusesToDestination_scaffold(string input1, int input2, int input3, int expectedResult) {
    Solution ss;
    vector<vector<int>> routes = stringTo2DArray<int>(input1);
    int actual = ss.numBusesToDestination(routes, input2, input3);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input1, input2, input3, expectedResult, actual);
    }
}


void distanceK_scaffold(string input1, int input2, int input3, string expectedResult) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input1);
    vector<int> actual = ss.distanceK(root, input2, input3);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input1, input2, input3, expectedResult, numberVectorToString(actual));
    }
}


void shortestAlternatingPaths_scaffold(int input1, string input2, string input3, string expectedResult) {
    Solution ss;
    vector<vector<int>> red_edges = stringTo2DArray<int>(input2);
    vector<vector<int>> blue_edges = stringTo2DArray<int>(input3);
    vector<int> actual = ss.shortestAlternatingPaths(input1, red_edges, blue_edges);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input2, input3, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input1, input2, input3, expectedResult, numberVectorToString(actual));
    }
}


void minPushBox_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<char>> grid = stringTo2DArray<char>(input);
    int actual = ss.minPushBox(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running minMutation tests:");
    TIMER_START(minMutation);
    minMutation_scaffold("AACCGGTT", "AACCGGTA", "[AACCGGTA]", 1);
    minMutation_scaffold("AACCGGTT", "AAACGGTA", "[AACCGGTA, AACCGCTA, AAACGGTA]", 2);
    minMutation_scaffold("AAAAACCC", "AACCCCCC", "[AAAACCCC, AAACCCCC, AACCCCCC]", 3);
    minMutation_scaffold("AAAAAAAA", "CCCCCCCC", "[AAAAAAAA,AAAAAAAC,AAAAAACC,AAAAACCC,AAAACCCC,AACACCCC,ACCACCCC,ACCCCCCC,CCCCCCCA]", -1);
    TIMER_STOP(minMutation);
    SPDLOG_WARN("minMutation tests {} ms", TIMER_MSEC(minMutation));

    SPDLOG_WARN("Running numBusesToDestination tests:");
    TIMER_START(numBusesToDestination);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 5, -1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 3, 1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 2, 5, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 1, 6, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 2, 3, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 6, 3, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 7, 1, 1);
    TIMER_STOP(numBusesToDestination);
    SPDLOG_WARN("numBusesToDestination tests {} ms", TIMER_MSEC(numBusesToDestination));

    SPDLOG_WARN("Running distanceK tests:");
    TIMER_START(distanceK);
    distanceK_scaffold("[3,5,1,6,2,0,8,null,null,7,4]", 5, 2, "[1,4,7]");
    distanceK_scaffold("[1]", 1, 2, "[]");
    distanceK_scaffold("[0,1,null,null,2,null,3,null,4]", 3, 0, "[3]");
    TIMER_STOP(distanceK);
    SPDLOG_WARN("distanceK tests {} ms", TIMER_MSEC(distanceK));

    SPDLOG_WARN("Running shortestAlternatingPaths tests:");
    TIMER_START(shortestAlternatingPaths);
    shortestAlternatingPaths_scaffold(3, "[[0,1],[1,2]]", "[]", "[0,1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[0,1]]", "[[2,1]]", "[0,1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[1,0]]", "[[2,1]]", "[0,-1,-1]");
    shortestAlternatingPaths_scaffold(3, "[[0,1]]", "[[1,2]]", "[0,1,2]");
    shortestAlternatingPaths_scaffold(3, "[[0,1],[0,2]]", "[[1,0]]", "[0,1,1]");
    TIMER_STOP(shortestAlternatingPaths);
    SPDLOG_WARN("shortestAlternatingPaths tests {} ms", TIMER_MSEC(shortestAlternatingPaths));

    SPDLOG_WARN("Running minPushBox tests:");
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

    grid = R"([[#,.,.,#,T,#,#,#,#],
               [#,.,.,#,.,#,.,.,#],
               [#,.,.,#,.,#,B,.,#],
               [#,.,.,.,.,.,.,.,#],
               [#,.,.,.,.,#,.,S,#],
               [#,.,.,#,.,#,#,#,#]])";
    minPushBox_scaffold(grid, 8);

    TIMER_STOP(minPushBox);
    SPDLOG_WARN("minPushBox tests {} ms", TIMER_MSEC(minPushBox));
}
