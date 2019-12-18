#include "leetcode.h"

using namespace std;

class Solution
{
public:
    int countComponents_dfs(int nodeCount, vector<vector<int>>& edges);
    int countComponents_unionFind(int nodeCount, vector<vector<int>>& edges);
};

int Solution::countComponents_unionFind(int nodeCount, vector<vector<int>>& edges)
{
    DSU dsu(nodeCount);
    for (auto& e: edges)
    {
        dsu.unionFunc(e[0], e[1]);
    }
    return dsu.groupCount();
}

int Solution::countComponents_dfs(int nodeCount, vector<vector<int>>& edges)
{
    vector<vector<int>> graph(nodeCount);
    for(auto& e: edges)
    {
        graph[e[0]].push_back(e[1]);
        graph[e[1]].push_back(e[0]);
    }

    unordered_set<int> seen;
    function<void(int)> dfs = [&] (int node)
    {
        if(!seen.count(node))
        {
            seen.emplace(node);
            for(auto& neighbor: graph[node])
                dfs(neighbor);
        }
    };

    int ans = 0;
    for (int i = 0; i < nodeCount; ++i)
    {
        if(seen.count(i)) continue;

        ++ans;

        for(auto& neighbor: graph[i])
        {
            dfs(neighbor);
        }
    }
    return ans;
}


int main()
{
    int nodeCount = 5;
    vector<vector<int>> edges {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4}
    };

    Solution ss;
    // cout << ss.countComponents_unionFind(nodeCount, edges) << "\n";
    cout << ss.countComponents_dfs(nodeCount, edges) << "\n";
}
