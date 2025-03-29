#include "leetcode.h"
#include "util/trip_farthest_insertion.hpp"

using namespace std;
using namespace osrm;

/* leetcode: 847, 864, 1298 */

class Solution {
public:
    int shortestPathLength(vector<vector<int>>& graph);
    int shortestPathAllKeys(vector<string>& grid);

private:
    int bruteForceTrip(vector<vector<int>>& distance_table, int node_count);
};

int Solution::shortestPathLength(vector<vector<int>>& graph) {
/*
    An *undirected* graph of N nodes (labelled 0, 1, 2, ..., N-1, N=graph.length) is given in *adjacency-list* notation.
    Return the length of the shortest path that visits every node. You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.
    Hint: Traveling Salesman Problem 
*/

    int node_count = graph.size();
    vector<vector<int>> distance_table(node_count, vector<int>(node_count, INT32_MAX));
    // initialize trivial cases
    for (int u=0; u<node_count; ++u) {
        distance_table[u][u] = 0;
        for (auto v: graph[u]) {
            distance_table[u][v] = 1;
            distance_table[v][u] = 1;
        }
    }
    // floyd-warshall algorithm
    for (int k=0; k<node_count; ++k) {
        for (int i=0; i<node_count; ++i) {
            for (int j=0; j<node_count; ++j) {
                if (distance_table[i][k] != INT32_MAX && distance_table[k][j] != INT32_MAX) {
                    distance_table[i][j] = std::min(distance_table[i][j], distance_table[i][k]+distance_table[k][j]);
                }
            }
        }
    }

    int maxWeight = INT32_MIN;
    for (auto& r: distance_table) {
        auto it = std::max_element(r.begin(), r.end());
        maxWeight = std::max(maxWeight, *it);
    }

    BOOST_ASSERT_MSG(maxWeight != INT32_MAX, "graph is not a strongly connected component");

#ifdef DEBUG
    for (int i=0; i<node_count; ++i) {
        std::cout << numberVectorToString(distance_table[i]) << std::endl;
    }
#endif

    // Taken from OSRM project
    if (node_count < 10) {
        // Time Limit Exceeded
        return bruteForceTrip(distance_table, node_count);
    } else {
        vector<int> node_order = scaffold::farthestInsertionTrip(node_count, distance_table);
#ifdef DEBUG
        cout << "Optimal plan: " << numberVectorToString(node_order) << endl;
#endif
        int length = 0;
        for (int i=1; i<node_count; ++i) {
            length += distance_table[node_order[i-1]][node_order[i]];
        }
        return length;
    }
}

int Solution::bruteForceTrip(vector<vector<int>>& distance_table, int node_count) {
    auto tripLengthForPlan = [&](vector<int>& node_order, int minLenth) {
        int length = 0;
        for (int i=1; i<node_count; ++i) {
            if (distance_table[node_order[i-1]][node_order[i]] == INT32_MAX) {
                return INT32_MAX;
            }
            length += distance_table[node_order[i-1]][node_order[i]];
            if (length >= minLenth) {
                break;
            }
        }
        return length;
    };

    int ans = INT32_MAX;
    vector<int> node_order(node_count);
    vector<int> plan = node_order;
    std::iota(node_order.begin(), node_order.end(), 0);
    // traversal all possible orders to visit all nodes, choose the best plan by path length
    do {
        int len = tripLengthForPlan(node_order, ans);
        if (len < ans) {
            plan = node_order;
            ans = len;
        }
    } while (std::next_permutation(node_order.begin(), node_order.end()));
    
#if defined(DEBUG_VERBOSITY)
    util::Log(logINFO) << "optimal plan: " << numberVectorToString(plan) << endl;
#endif
    return ans;    
}

int Solution::shortestPathAllKeys(vector<string>& grid) {
/*
    We are given a 2-dimensional grid. "." is an empty cell, "#" is a wall, "@" is the starting point, ("a", "b", …) are keys, and ("A", "B", …) are locks.

    We start at the starting point, and one move consists of walking one space in one of the 4 cardinal directions.
    We cannot walk outside the grid, or walk into a wall. If we walk over a key, we pick it up. We can’t walk over a lock unless we have the corresponding key.

    For some 1 <= K <= 6, there is exactly one lowercase and one uppercase letter of the first K letters 
    of the English alphabet in the grid. This means that there is exactly one key for each lock, 
    and one lock for each key; and also that the letters used to represent the keys and locks 
    were chosen in the same order as the English alphabet.

    Return the lowest number of moves to acquire all keys. If it’s impossible, return -1.
*/

    auto is_key = [] (int letter) {
        return 'a' <= letter && letter <= 'z';
    };
    auto is_lock = [] (int letter) {
        return 'A' <= letter && letter <= 'Z';
    };

    // 1. find start, keys, locks positions
    using Coordinate = std::pair<int, int>; // <row, column>
    Coordinate coors[128];
    int keys = 0;
    int rows = grid.size();
    int columns = grid[0].size();
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; ++j) {
            int letter = grid[i][j];
            if (letter == '@' || is_key(letter) || is_lock(letter)){
                keys++;
                coors[letter].first = i;
                coors[letter].second = j;
            }
        }
    }

    keys /= 2; // number of the pairs of key/lock
    auto route_length = [&] () {
        int steps = 0;
        set<int> visited_locks;
        set<int> visited_keys;
        const int diff = 'a' - 'A';
        Coordinate start = coors[(int)'@'];
        set<Coordinate> visited; visited.insert(start);
        queue<Coordinate> q; q.push(start);
        while (!q.empty()) {
            if ((int)visited_keys.size() == keys) {
                return steps;
            }
            for (int k=q.size(); k!=0; --k) {
                auto u = q.front(); q.pop();
                for (auto& d: directions) {
                    int nr = u.first + d.first;
                    int nc = u.second + d.second;
                    // out of boundary or ecounter walls
                    if (nr<0 || nr >=rows ||
                        nc<0 || nc >=columns ||
                        grid[nr][nc] == '#') {
                        continue;
                    }
                    // already visited
                    if (visited.count({nr, nc}) == 1) {
                        continue;
                    }
                    int letter = grid[nr][nc];
                    if (letter == '.') { // empty cells
                        q.emplace(nr, nc);
                        visited.emplace(nr, nc);
                    } else if (is_key(letter)) { // keys
                        visited_keys.insert(letter);
                        q.emplace(nr, nc);
                        visited.emplace(nr, nc);
                        // after we got the key we may pass the lock next time
                        if (visited_locks.count(letter-diff) == 1) {
                            q.push(coors[letter-diff]);
                            visited.insert(coors[letter-diff]);
                        }  
                    } else if (is_lock(letter)) { // locks
                        if (visited_keys.count(letter+diff) == 1) { // we can pass lock only if we have alread got its key
                            q.emplace(nr, nc);
                            visited.emplace(nr, nc);
                        } else {
                            visited_locks.insert(letter);
                        }
                    }
                }
            }
            ++steps;
        }
        return INT32_MAX;
    };
    int ans = route_length();
    return ans == INT32_MAX ? -1 : ans;
}


void shortestPathLength_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.shortestPathLength(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void shortestPathAllKeys_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<string> graph = stringTo1DArray<string>(input);
    int actual = ss.shortestPathAllKeys(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running shortestPathLength tests:");
    TIMER_START(shortestPathLength);
    shortestPathLength_scaffold("[[1,2,3],[0],[0],[0]]", 4);
    shortestPathLength_scaffold("[[1],[0,2,4],[1,3,4],[2],[1,2]]", 4);
    shortestPathLength_scaffold("[[2,6],[2,3],[0,1],[1,4,5,6,8],[3,9,7],[3],[3,0],[4],[3],[4]]", 12);
    TIMER_STOP(shortestPathLength);
    SPDLOG_WARN("shortestPathLength tests use {} ms", TIMER_MSEC(shortestPathLength));

    SPDLOG_WARN("Running shortestPathAllKeys tests:");
    TIMER_START(shortestPathAllKeys);
    shortestPathAllKeys_scaffold("[@.a.#, ###.#, b.A.B]", 8);
    shortestPathAllKeys_scaffold("[@.a.., ###.#, b.A.B]", 8);
    shortestPathAllKeys_scaffold("[@..aA, ..B#., ....b]", 6);
    shortestPathAllKeys_scaffold("[@Aa]", -1);
    shortestPathAllKeys_scaffold("[@...a, .###A, b.BCc]", 6);
    TIMER_STOP(shortestPathAllKeys);
    SPDLOG_WARN("shortestPathAllKeys tests use {} ms", TIMER_MSEC(shortestPathAllKeys));
}
