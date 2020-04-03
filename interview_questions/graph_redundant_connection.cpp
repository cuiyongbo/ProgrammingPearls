#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 684, 685, 1319 */

class DisjointSet
{
public:
    DisjointSet(int n)
    {
        m_rank.resize(n+1, 0);
        m_parent.resize(n+1, 0);
        std::iota(m_parent.begin(), m_parent.end(), 0);
    }

    int find(int x)
    {
        if(m_parent[x] != x)
        {
            m_parent[x] = find(m_parent[x]);
        }
        return m_parent[x];
    }

    bool unionFunc(int x, int y)
    {
        int px = find(x);
        int py = find(y);

        if(px == py)
            return false; // cycle detected

        if(m_rank[px] > m_rank[py])
        {
            m_parent[py] = px;
        }
        else
        {
            m_parent[px] = py;
            if(m_rank[px] == m_rank[py])
                ++m_rank[py];
        }
        return true;
    }

private:
    vector<int> m_parent;
    vector<int> m_rank;
};

class Solution
{
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges);
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges);
};

vector<int> Solution::findRedundantConnection(vector<vector<int>>& edges)
{
    /*
        In this problem, a tree is an undirected graph that is connected and has no cycles.
        The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N),
        with one additional edge added. The added edge has two different vertices chosen from 1 to N,
        and was not an edge that already existed.

        The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] with u < v,
        that represents an undirected edge connecting nodes u and v.

        Return an edge that can be removed so that the resulting graph is a tree of N nodes.
        If there are multiple answers, return the answer that occurs last in the given 2D-array.
        The answer edge [u, v] should be in the same format, with u < v.
    */

    DisjointSet s(edges.size());
    for(const auto& e: edges)
    {
        if(!s.unionFunc(e[0], e[1]))
            return e;
    }
    return {};
}

vector<int> Solution::findRedundantDirectedConnection(vector<vector<int>>& edges)
{
    /*
        In this problem, a rooted tree is a directed graph such that,
        there is exactly one node (the root) for which all other nodes are descendants of this node,
        plus every node has exactly one parent, except for the root node which has no parents.

        The given input is a directed graph that started as a rooted tree with N nodes (with distinct values 1, 2, …, N),
        with one additional directed edge added. The added edge has two different vertices chosen from 1 to N,
        and was not an edge that already existed.

        The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v]
        that represents a directed edge connecting nodes u and v, where u is a parent of child v.

        Return an edge that can be removed so that the resulting graph is a rooted tree of N nodes.
        If there are multiple answers, return the answer that occurs last in the given 2D-array.
    */

    // case 1: no duplicate parents, revert to findRedundantConnection
    // case 2: no cycle, a node has duplicate parents [u1, u2], return the latter
    // case 3: both, return [u1, v] or [u2, v] that creates the loop

    int nodeCount = edges.size();
    vector<vector<int>> graph(nodeCount+1, vector<int>());

    vector<int> duplicateParentEdge;
    bool duplicateParentFound = false;

    for(const auto& e: edges)
    {
        graph[e[1]].push_back(e[0]);
        if(graph[e[1]].size() > 1)
        {
            duplicateParentEdge = e;
            duplicateParentFound = true;
            break;
        }
    }

    vector<int> cycleEdge;
    bool cycleFound = false;
    DisjointSet dsu(nodeCount);
    for(const auto& e: edges)
    {
        if(!dsu.unionFunc(e[0], e[1]))
        {
            cycleFound = true;
            cycleEdge = e;
            break;
        }
    }

    if(cycleFound && duplicateParentFound)
    {
        int v = duplicateParentEdge[1];
        if(dsu.find(graph[v][0]) == dsu.find(v))
        {
            return {graph[v][0], v};
        }
        else
        {
            return {graph[v][1], v};
        }
    }
    else if(cycleFound)
    {
        return cycleEdge;
    }
    else if(duplicateParentFound)
    {
        return duplicateParentEdge;
    }
    else
    {
        return {};
    }
}

void findRedundantConnection_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    vector<int> actual = ss.findRedundantConnection(graph);
    vector<int> expected = stringToIntegerVector(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << "," << expectedResult << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", acutal: " << intVectorToString(actual);
    }
}

void findRedundantDirectedConnection_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    vector<int> actual = ss.findRedundantDirectedConnection(graph);
    vector<int> expected = stringToIntegerVector(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << "," << expectedResult << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", acutal: " << intVectorToString(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running findRedundantConnection tests:";
    TIMER_START(findRedundantConnection);
    findRedundantConnection_scaffold("[[1,2], [1,3], [2,3]]", "[2,3]");
    findRedundantConnection_scaffold("[[1,2], [2,3], [3,4], [1,4], [1,5]]", "[1,4]");
    TIMER_STOP(findRedundantConnection);
    util::Log(logESSENTIAL) << "findRedundantConnection using " << TIMER_MSEC(findRedundantConnection) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findRedundantDirectedConnection tests:";
    TIMER_START(findRedundantDirectedConnection);
    findRedundantDirectedConnection_scaffold("[[1,2], [2,3], [3,4], [4,1], [1,5]]", "[4,1]"); // no duplicate parents, revert to findRedundantConnection
    findRedundantDirectedConnection_scaffold("[[1,2], [1,3], [2,3]]", "[2,3]"); // a node has duplicate parents [u, v], return the latter
    findRedundantDirectedConnection_scaffold("[[1,2], [2,3], [3,4], [1,4], [1,5]]", "[1,4]");
    findRedundantDirectedConnection_scaffold("[[1,2], [2,3], [3,4], [4,2], []]", "[4, 2]");
    findRedundantDirectedConnection_scaffold("[[2,1], [3,1], [4,2], [1,4]", "[2,1]"); // both, return [u1, v] or [u2, v] that creates the loop
    findRedundantDirectedConnection_scaffold("[[3,1], [2,1], [4,2], [1,4]", "[2,1]");
    TIMER_STOP(findRedundantDirectedConnection);
    util::Log(logESSENTIAL) << "findRedundantDirectedConnection using " << TIMER_MSEC(findRedundantDirectedConnection) << " milliseconds";


}
