#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 743, 787, 882, 1334 */

class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int N, int K);
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K);
    int reachableNodes(vector<vector<int>>& edges, int M, int N);
    int findTheCity(int n, vector<vector<int>>& edges, int t);
};


/*
    There are N network nodes, labelled 1 to N. Given times, a list of travel times as *directed* edges times[i] = (u, v, w),
    where u is the source node, v is the target node, and w is the time it takes for a signal to travel from source to target.
    Now we send a signal from a certain node K. How long will it take for all nodes to receive the signal? If it is impossible, return -1.
    Hint: perform dijkstra search on the graph
*/
int Solution::networkDelayTime(vector<vector<int>>& times, int N, int K) {

{ // refined solution
    // build graph
    using element_t = std::pair<int, int>; // node id, weight
    vector<vector<element_t>> graph(N);
    for (auto p: times) {
        graph[p[0]-1].emplace_back(p[1]-1, p[2]);
    }
    auto cmp_by_weight = [] (const element_t& l, const element_t& r) {
        return l.second > r.second;
    };
    std::map<int,int> visited; // node u, time taking from K to node u
    std::priority_queue<element_t, vector<element_t>, decltype(cmp_by_weight)> pq(cmp_by_weight); // min_heap ordered by weight
    // initialization
    visited[K-1] = 0;
    pq.emplace(K-1, 0);
    while (!pq.empty()) {
        auto u = pq.top(); pq.pop();
        for (auto& v: graph[u.first]) {
            if (visited.count(v.first) == 0 || // not visited
                visited[v.first] > u.second+v.second) { // find a route costing less time
                visited[v.first] = u.second+v.second;
                pq.emplace(v.first, visited[v.first]);
            }
        }
    }
    int ans = -1;
    if (visited.size() == N) {
        for (auto p: visited) {
            ans = max(ans, p.second);
        }
    }
    return ans;
}

{ // naive solution
    typedef pair<int,int> element_t; // target_node, cost
    vector<vector<element_t>> graph(N+1);
    for (auto& t: times) {
        graph[t[0]].emplace_back(t[1], t[2]);
    }
    // Be cautious when initialzing preconditions!
    vector<bool> visited(N+1, false); visited[K] = true;
    vector<int> cost(N+1, INT32_MAX); cost[K] = 0;
    auto cmp_by_cost = [&] (const element_t& l, const element_t& r) {
        return l.second > r.second;
    };
    priority_queue<element_t, vector<element_t>, decltype(cmp_by_cost)> pq(cmp_by_cost);
    pq.emplace(K, 0);
    while (!pq.empty()) {
        auto u = pq.top(); pq.pop();
        for (auto& v: graph[u.first]) {
            if (!visited[v.first] || (cost[v.first] > u.second+v.second)) {
                visited[v.first] = true;
                cost[v.first] = u.second+v.second;
                pq.emplace(v.first, cost[v.first]);
            }
        }
    }
    //print_vector(cost);
    int ans = *(max_element(cost.begin()+1, cost.end()));
    return ans==INT32_MAX ? -1 : ans;
}

}


/*
    There are n cities connected by m flights. Each flight [u, v, w] starts from city u and arrives at v with a price w.
    Now given all the cities (labelled 0 to N-1) and flights, together with starting city `src` and the destination `dst`,
    your task is to find the cheapest price from `src` to `dst` with at most `K` stops. If there is no such route, return -1.
    hint: dijkstra algorithm
*/
int Solution::findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
    // build a directed graph
    using element_t = std::pair<int, int>;
    vector<vector<element_t>> graph(n); // dest, time
    for (const auto& t: flights) {
        graph[t[0]].emplace_back(t[1], t[2]);
    }
    int steps = 0;
    set<bool> visited; visited.insert(src);
    vector<int> cost(n, INT32_MAX); cost[src] = 0;
    queue<element_t> q; q.emplace(src, 0);
    while (steps<=K && !q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            // DON'T uncomment following test, the route with the least steps may not be the cheapest route
            //if (u.first == dst) {
            //    return costs[u.first];
            //}
            for (auto& v: graph[u.first]) {
                if (visited.count(v.first) == 0 || // not visited
                        cost[v.first] > u.second + v.second) { // find a cheaper route
                    visited.insert(v.first);
                    cost[v.first] = u.second + v.second;
                    q.emplace(v.first, cost[v.first]);
                }
            }
        }
        ++steps;
    }
    return cost[dst] == INT32_MAX ? -1 : cost[dst];
}


/*
    Starting with an *undirected* graph with N nodes labelled 0 to N-1, subdivisions are made to some of the edges.
    The graph is given as follows: edges[k] is a list of integer pairs (i, j, n) such that (i, j) is an edge of the original graph,
    and n is the total number of new nodes on that edge. Then the edge (i, j) is deleted from the original graph, 
    n new nodes (x_1, x_2, ..., x_n) are added to the original graph, and n+1 new edges (i, x_1), (x_1, x_2), (x_2, x_3), ..., (x_{n-1}, x_n), (x_n, j) are added to the original graph.
    Now, you start at node 0 from the original graph, and in each move, you travel along one edge. Return how many nodes you can reach in at most M moves.
*/
int Solution::reachableNodes(vector<vector<int>>& edges, int M, int N) {

if (0) {
    using element_t = std::pair<int, int>;
    vector<vector<element_t>> graph(N);
    for (auto& e: edges) {
        graph[e[0]].emplace_back(e[1], e[2]+1);
        graph[e[1]].emplace_back(e[0], e[2]+1);
    }
    auto cmp = [&] (const element_t& l, const element_t& r) {
        return l.second > r.second;
    };
    int ans = 0;
    vector<int> distance_table(N, INT32_MAX); distance_table[0] = 0; // distance_table[u] means the number of visited nodes starting from node u
    priority_queue<element_t, vector<element_t>, decltype(cmp)> pq(cmp); // node, number of visited nodes
    pq.emplace(0, 0);
    while (!pq.empty()) {
        auto t = pq.top(); pq.pop();
        for (auto& p: graph[t.first]) {
            if ((t.second+p.second) < distance_table[p.first]) {
                distance_table[p.first] = p.second+t.second;
                pq.emplace(p.first, distance_table[p.first]);
            }
        }
    }
    for (auto d: distance_table) {
        if (d<=M) {
            ans++;
        }
    }
    for (auto& e: edges) {
        int cnt = e[2];
        int a = std::min(cnt, std::max(0, M-distance_table[e[1]]));
        int b = std::min(cnt, std::max(0, M-distance_table[e[0]]));
        ans += std::min(cnt, a+b);
    }
    return ans;
}

{ // naive bfs version, Time Limit Exceeded
    // build graph
    map<int, vector<int>> graph;
    int new_node_id = N;
    for (auto& e: edges) {
        int start = e[0];
        int mid = 0;
        for (int i=0; i<e[2]; ++i) {
            mid = new_node_id++;
            graph[start].emplace_back(mid);
            graph[mid].emplace_back(start);
            start = mid;
        }
        graph[start].emplace_back(e[1]);
        graph[e[1]].emplace_back(start);
    }
    int ans = 0;
    int steps = 0;
    set<int> visited; visited.emplace(0);
    queue<int> q; q.emplace(0);
    while (!q.empty() && steps<=M) {
        ans += q.size(); // number of nodes visited at each step
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
    return ans;
}

}


/*
    There are n cities labelled from 0 to N-1. Given the array edges where edges[i] = [u, v, weight]
    represents a *bidirectional* and weighted edge between cities u and v, and distance_threshold.

    Return the city with the smallest number of cities that are reachable through some path and whose distance is 
    at most distance_threshold, If there are multiple such cities, return the city with the greatest index.

    Notice that the distance of a path connecting cities i and j is equal to the sum of the edges' weights along that path.

    Hint: Floyd-Warshall algorithm
*/
int Solution::findTheCity(int N, vector<vector<int>>& edges, int distance_threshold) {
    vector<vector<int>> distance_table(N, vector<int>(N, INT32_MAX));
    // initialize trivial cases
    for (auto& e: edges) {
        distance_table[e[0]][e[1]] = e[2];
        distance_table[e[1]][e[0]] = e[2];
    }
    // runtime complexity: O(n^3)
    for (int k=0; k<N; ++k) {
        // use k as waypoint
        distance_table[k][k] = 0; // trivial cases
        for (int i=0; i<N; ++i) {
            for (int j=0; j<N; ++j) {
                if (distance_table[i][k] != INT32_MAX && distance_table[k][j] != INT32_MAX) {
                    distance_table[i][j] = min(distance_table[i][j], distance_table[i][k] + distance_table[k][j]);
                }
            }
        }
    }
    int city = INT32_MIN;
    int city_num = INT32_MAX;
    for (int r=0; r<N; ++r) {
        int cur = std::accumulate(distance_table[r].begin(), distance_table[r].end(), 0,
                                    [&](int s, int d) {return d<=distance_threshold ? s+1 : s;});
        if (city_num >= cur) {
            city_num = cur;
            city = r;
        }
    }
    return city;
}


void networkDelayTime_scaffold(string input, int N, int K, int expectedResult) {
    Solution ss;
    vector<vector<int>> times = stringTo2DArray<int>(input);
    int actual = ss.networkDelayTime(times, N, K);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input, N, K, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input, N, K, expectedResult, actual);
    }
}


void findCheapestPrice_scaffold(int N, string input, int src, int dst, int K,  int expectedResult) {
    Solution ss;
    vector<vector<int>> flights = stringTo2DArray<int>(input);
    int actual = ss.findCheapestPrice(N, flights, src, dst, K);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, {}, {}, expectedResult={}) passed", N, input, src, dst, K, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, {}, {}, expectedResult={}) failed, actual={}", N, input, src, dst, K, expectedResult, actual);
    }
}


void reachableNodes_scaffold(string input, int M, int N, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.reachableNodes(graph, M, N);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input, M, N, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input, M, N, expectedResult, actual);
    }
}


void findTheCity_scaffold(string input, int N, int distance_threshold, int expectedResult) {
    Solution ss;
    vector<vector<int>> edges = stringTo2DArray<int>(input);
    int actual = ss.findTheCity(N, edges, distance_threshold);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input, N, distance_threshold, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual={}", input, N, distance_threshold, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running networkDelayTime tests:");
    TIMER_START(networkDelayTime);
    networkDelayTime_scaffold("[[2,1,1],[2,3,1],[3,4,1]]", 4, 2, 2);
    networkDelayTime_scaffold("[[1,2,1],[2,3,2],[1,3,4]]", 3, 1, 3);
    networkDelayTime_scaffold("[[1,2,1]]", 2, 2, -1);
    TIMER_STOP(networkDelayTime);
    SPDLOG_WARN("networkDelayTime tests use {} ms", TIMER_MSEC(networkDelayTime));

    SPDLOG_WARN("Running findCheapestPrice tests:");
    TIMER_START(findCheapestPrice);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 1, 200);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 0, 500);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 10, 200);
    TIMER_STOP(findCheapestPrice);
    SPDLOG_WARN("findCheapestPrice tests use {} ms", TIMER_MSEC(findCheapestPrice));

    SPDLOG_WARN("Running reachableNodes tests:");
    TIMER_START(reachableNodes);
    reachableNodes_scaffold("[[0,1,0],[0,2,0],[1,2,0]]", 6, 3, 3);
    reachableNodes_scaffold("[[0,1,10],[0,2,1],[1,2,2]]", 6, 3, 13);
    reachableNodes_scaffold("[[0,1,4],[1,2,6],[0,2,8],[1,3,1]]", 10, 4, 23);
    reachableNodes_scaffold("[[1,2,5],[0,3,3],[1,3,2],[2,3,4],[0,4,1]]", 7, 5, 13);
    reachableNodes_scaffold("[[0,3,8],[0,1,4],[2,4,3],[1,2,0],[1,3,9],[0,4,7],[3,4,9],[1,4,4],[0,2,7],[2,3,1]]", 8, 5, 40);
    TIMER_STOP(reachableNodes);
    SPDLOG_WARN("reachableNodes tests use {} ms", TIMER_MSEC(reachableNodes));

    SPDLOG_WARN("Running findTheCity tests:");
    TIMER_START(findTheCity);
    findTheCity_scaffold("[[0,1,3],[1,2,1],[1,3,4],[2,3,1]]", 4, 4, 3);
    findTheCity_scaffold("[[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]]", 5, 2, 0);
    TIMER_STOP(findTheCity);
    SPDLOG_WARN("findTheCity tests use {} ms", TIMER_MSEC(findTheCity));
}
