#include "leetcode.h"

using namespace std;

class Solution
{
public:
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections);
};

vector<vector<int>> Solution::criticalConnections(int n, vector<vector<int>>& connections)
{
    vector<vector<int>> graph(n);
    for(auto& p: connections)
    {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    int t = 0;
    vector<vector<int>> ans;
    vector<int> ts(n, -1);
    function<int(int, int)> tarjan = [&](int u, int p)
    {
        int min_u = ts[u] = t++;
        for(auto v: graph[u])
        {
            if(ts[v] == -1)
            {
                int min_v = tarjan(v, u);
                min_u = min(min_u, min_v);
                if(ts[u] < min_v)
                    ans.push_back({u, v});
            }
            else if(v != p)
            {
                min_u = min(min_u, ts[v]);
            }
        }
        return min_u;
    };

    tarjan(0, -1);
    return ans;
}

int main()
{
    Solution ss;
    vector<vector<int>> connections = {{0,1},{1,2},{2,0},{1,3}};
    vector<vector<int>> result = ss.criticalConnections(4, connections);
    for(auto& p : result)
    {
        cout << "(" << p[0] << ", " << p[1] << ")\n";
    }
    return 0;
}
