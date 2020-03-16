#include "leetcode.h"

using namespace std;

/*
Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into 
two independent subsets A and B such that every edge in the graph has 
one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j 
for which the edge between nodes i and j exists.  Each node is an integer 
between 0 and graph.length - 1. There are no self edges or parallel edges: 
graph[i] does not contain i, and it doesn't contain any element twice.
*/


class Solution
{
public:
    bool isBipartite(vector<vector<int>>& graph);
};

bool Solution::isBipartite(vector<vector<int>>& graph)
{
    int n = (int)graph.size();
    vector<int> colors(n, 0);
    colors[0] = 1;
    queue<int> q;
    q.push(0);
    while(!q.empty())
    {
        auto u = q.front();
        q.pop();

        for(auto v: graph[u])
        {
            if(colors[v] == 0)
            {
                q.push(v);
                colors[v] = colors[u] == 1 ? 2 : 1;
            }
            else if(colors[u] == colors[v])
            {
                return false;
            }
        }
    }
    return true;
}

int main()
{
    // vector<vector<int>> graph = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    vector<vector<int>> graph = {{1,2,3}, {0,2}, {0,1,3}, {0,2}};
    Solution ss;
    bool result = ss.isBipartite(graph);
    if(result)
    {
        cout << "graph is bipartite\n";
    }
    else
    {
        cout << "graph is not bipartite\n";
    }
    return 0;
}
