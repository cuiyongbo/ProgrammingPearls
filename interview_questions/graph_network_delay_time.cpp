#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 743, 787, 882, 1334 */

class Solution
{
public:
    int networkDelayTime(vector<vector<int>>& times, int N, int K);
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K);
    int reachableNodes(vector<vector<int>>& edges, int M, int N);
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial);
    int findTheCity(int n, vector<vector<int>>& edges, int t);
};

int Solution::networkDelayTime(vector<vector<int>>& times, int N, int K)
{
    /*
        There are N network nodes, labelled 1 to N.
        Given times, a list of travel times as directed edges times[i] = (u, v, w),
        where u is the source node, v is the target node, and w is the time it takes 
        for a signal to travel from source to target.

        Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal?
        If it is impossible, return -1.
    */

    vector<vector<pair<int, int>>> graph(N+1); // dest, time
    for(const auto& t: times)
    {
        graph[t[0]].push_back({t[1], t[2]});
    }

    int ans = -1;
    set<int> visited;
    queue<pair<int, int>> q;
    q.push({K, 0});
    while(!q.empty())
    {
        int size = (int)q.size();
        for(int i=0; i<size; ++i)
        {
            const auto& u = q.front(); q.pop();
            visited.emplace(u.first);
            ans = std::max(ans, u.second);

            for(const auto& v: graph[u.first])
            {
                if(visited.count(v.first) != 0)
                    continue;
                
                q.push({v.first, u.second + v.second});
            }
        }
    }

    return N == (int)visited.size() ? ans : -1;
}

int Solution::findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K)
{
    /*
        There are n cities connected by m flights. 
        Each flight [u, v, w] starts from city u and arrives at v with a price w.

        Now given all the cities (labelled 0 to N-1) and flights, together with starting city `src` and the destination `dst`,
        your task is to find the cheapest price from `src` to `dst` with up to `K` stops. 
        If there is no such route, output -1.
    */

    vector<vector<pair<int, int>>> graph(n); // dest, time
    for(const auto& t: flights)
    {
        graph[t[0]].push_back({t[1], t[2]});
    }

    vector<int> distanceTable(n, INT32_MAX);
    distanceTable[src] = 0;

    int steps = 0;
    queue<pair<int, int>> q;
    q.push({src, 0});
    while(steps <= K && !q.empty())
    {
        int size = (int)q.size();
        for(int i=0; i<size; ++i)
        {
            auto u = q.front(); q.pop();
            for(const auto& v: graph[u.first])
            {
                if(distanceTable[v.first] > u.second + v.second)
                {
                    distanceTable[v.first] = u.second + v.second;
                    q.push({v.first, distanceTable[v.first]});
                }
            }
        }
        ++steps;
    }

    return distanceTable[dst] == INT32_MAX ? -1 : distanceTable[dst];
}

int Solution::reachableNodes(vector<vector<int>>& edges, int M, int N)
{
    /*
        Starting with an undirected graph with N nodes labelled 0 to N-1, subdivisions are made to some of the edges.
        The graph is given as follows: edges[k] is a list of integer pairs (i, j, n) such that (i, j) is an edge of the original graph,
        and n is the total number of new nodes on that edge. Then, the edge (i, j) is deleted from the original graph, 
        n new nodes (x_1, x_2, ..., x_n) are added to the original graph,
        and n+1 new edges (i, x_1), (x_1, x_2), (x_2, x_3), ..., (x_{n-1}, x_n), (x_n, j) are added to the original graph.
        Now, you start at node 0 from the original graph, and in each move, you travel along one edge.
        Return how many nodes you can reach in at most M moves.
    */

    // I can't understand the description

    return -1;
}

int Solution::minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial)
{
    /*
        In a network of nodes, each node i is directly connected to another node j if and only if graph[i][j] = 1.

        Some nodes initial are initially infected by malware. Whenever two nodes are directly connected 
        and at least one of those two nodes is infected by malware, both nodes will be infected by malware.  
        This spread of malware will continue until no more nodes can be infected in this manner.

        Suppose M(initial) is the final number of nodes infected with malware in the entire network, 
        after the spread of malware stops.

        We will remove one node from the initial list.  Return the node that if removed, would minimize M(initial).
        If multiple nodes could be removed to minimize M(initial), return such a node with the smallest index. 

        Note that if a node was removed from the initial list of infected nodes, 
        it may still be infected later as a result of the malware spread.
    */

    int nodeCount = graph.size();
    DisjointSet dsu(nodeCount);
    for(int i=0; i<nodeCount; ++i)
    {
        for(int j=0; j<i; ++j)
        {
            if(graph[i][j]) 
                dsu.unionFunc(i, j);
        }
    }

    map<int, int> groupCountMap;
    for(int i=0; i<nodeCount; ++i)
    {
        groupCountMap[dsu.find(i)]++;
    }

    map<int, pair<int, int>> initGrp; // groupId, nodeCount, minGroupId
    for(const auto& u: initial)
    {
        int g = dsu.find(u);
        if(initGrp.count(g) == 0)
        {
            initGrp[g] = {1, u};
        }
        else
        {
            initGrp[g].first++;
            initGrp[g].second = std::min(u, initGrp[g].second);
        }
    }

    int lastMinus = 0;
    int ans = nodeCount;
    for(const auto& it: initGrp)
    {
        int minus = it.second.first == 1 ? groupCountMap[it.first] : 0;
        if(minus > lastMinus)
        {
            lastMinus = minus;
            ans = it.second.second;
        }
        else if(minus == lastMinus)
        {
            ans = std::min(ans, it.second.second);
        }
    }
    return ans;
}

int Solution::findTheCity(int N, vector<vector<int>>& edges, int distanceThreshold)
{
    /*
        There are n cities numbered from 0 to N-1. Given the array edges where edges[i] = [fromi, toi, weighti]
        represents a *bidirectional* and weighted edge between cities fromi and toi, and given the integer distanceThreshold.

        Return the city with the smallest number of cities that are reachable through some path and whose distance is 
        at most distanceThreshold, If there are multiple such cities, return the city with the greatest number.

        Notice that the distance of a path connecting cities i and j is equal to the sum of the edges’ weights along that path.        

        Hint: Floyd-Warshall algorithm
    */

    vector<vector<int>> distanceTable(N, vector<int>(N, INT32_MAX));
    for(int i=0; i<N; ++i) distanceTable[i][i] = 0;

    for(const auto& e: edges)
    {
        distanceTable[e[0]][e[1]] = e[2];
        distanceTable[e[1]][e[0]] = e[2];
    }

    for(int k=0; k<N; ++k)
    {
        for(int i=0; i<N; ++i)
        {
            for(int j=0; j<N; ++j)
            {
                if(distanceTable[i][k] != INT32_MAX && distanceTable[k][j] != INT32_MAX)
                {
                    distanceTable[i][j] = std::min(distanceTable[i][j], distanceTable[i][k]+distanceTable[k][j]);
                }
            }
        }
    }

    // for debug
    // for(const auto& r: distanceTable)
    // {
    //     cout << numberVectorToString(r) << endl;
    // }

    int ans = N;
    int reachableCities = N;
    for(int r=0; r<N; ++r)
    {
        int cur = std::accumulate(distanceTable[r].begin(), distanceTable[r].end(), 0, 
                                    [&](int s, int d) { return d<=distanceThreshold ? s+1 : s;});
        
        if(cur <= reachableCities)
        {
            ans = r;
            reachableCities = cur;
        }
    }
    return ans;
}

void networkDelayTime_scaffold(string input, int N, int K, int expectedResult)
{
    Solution ss;
    vector<vector<int>> times = stringTo2DArray_t<int>(input);
    int actual = ss.networkDelayTime(times, N, K);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << N << ", " << K << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << N << ", " << K << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findCheapestPrice_scaffold(int N, string input, int src, int dst, int K,  int expectedResult)
{
    Solution ss;
    vector<vector<int>> flights = stringTo2DArray_t<int>(input);
    int actual = ss.findCheapestPrice(N, flights, src, dst, K);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << N << ", " << K << ", " << src << ", " << dst << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << N << ", " << K << ", " << src << ", " << dst << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void reachableNodes_scaffold(string input, int M, int N, int expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray_t<int>(input);
    int actual = ss.reachableNodes(graph, M, N);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << M << ", " << N << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << M << ", " << N << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void minMalwareSpread_scaffold(string input1, string input2, int expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray_t<int>(input1);
    vector<int> initial = stringTo1DArray_t<int>(input2);
    int actual = ss.minMalwareSpread(graph, initial);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findTheCity_scaffold(string input, int N, int distanceThreshold, int expectedResult)
{
    Solution ss;
    vector<vector<int>> edges = stringTo2DArray_t<int>(input);
    int actual = ss.findTheCity(N, edges, distanceThreshold);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << N << ", " << distanceThreshold << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << N << ", " << distanceThreshold << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running networkDelayTime tests:";
    TIMER_START(networkDelayTime);
    networkDelayTime_scaffold("[[2,1,1],[2,3,1],[3,4,1]]", 4, 2, 2);
    TIMER_STOP(networkDelayTime);
    util::Log(logESSENTIAL) << "networkDelayTime using " << TIMER_MSEC(networkDelayTime) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findCheapestPrice tests:";
    TIMER_START(findCheapestPrice);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 1, 200);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 0, 500);
    findCheapestPrice_scaffold(3, "[[0,1,100],[1,2,100],[0,2,500]]", 0, 2, 10, 200);
    TIMER_STOP(findCheapestPrice);
    util::Log(logESSENTIAL) << "findCheapestPrice using " << TIMER_MSEC(findCheapestPrice) << " milliseconds";

    util::Log(logESSENTIAL) << "Running reachableNodes tests:";
    TIMER_START(reachableNodes);
    reachableNodes_scaffold("[[0,1,10],[0,2,1],[1,2,2]]", 6, 3, 13);
    reachableNodes_scaffold("[[0,1,4],[1,2,6],[0,2,8],[1,3,1]]", 10, 4, 23);
    TIMER_STOP(reachableNodes);
    util::Log(logESSENTIAL) << "reachableNodes using " << TIMER_MSEC(reachableNodes) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minMalwareSpread tests:";
    TIMER_START(minMalwareSpread);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1]", 0);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1,2]", 2);
    minMalwareSpread_scaffold("[[1,0,0],[0,1,0],[0,0,1]]", "[0,2]", 0);
    minMalwareSpread_scaffold("[[1,1,1],[1,1,1],[1,1,1]]", "[1,2]", 1);
    minMalwareSpread_scaffold("[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,0,0],[0,0,1,0,1,0],[0,0,0,0,0,1]]", "[4,3]", 4);
    TIMER_STOP(minMalwareSpread);
    util::Log(logESSENTIAL) << "minMalwareSpread using " << TIMER_MSEC(minMalwareSpread) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findTheCity tests:";
    TIMER_START(findTheCity);
    findTheCity_scaffold("[[0,1,3],[1,2,1],[1,3,4],[2,3,1]]", 4, 4, 3);
    findTheCity_scaffold("[[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]]", 5, 2, 0);
    TIMER_STOP(findTheCity);
    util::Log(logESSENTIAL) << "findTheCity using " << TIMER_MSEC(findTheCity) << " milliseconds";
}
