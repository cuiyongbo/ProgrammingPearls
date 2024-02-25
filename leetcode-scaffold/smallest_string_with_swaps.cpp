#include "leetcode.h"

using namespace std;

class Solution {
public:
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs);
    string smallestStringWithSwaps_unionFound(string s, vector<vector<int>>& pairs);
};

string Solution::smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
{
    int len = s.length();
    vector<vector<int>> graph(len);
    for(auto& p: pairs)
    {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    string tmp;
    vector<int> idx;
    unordered_set<int> seen;
    function<void(int)> dfs = [&] (int cur)
    {
        if(seen.count(cur))
            return;

        seen.insert(cur);
        idx.push_back(cur);
        tmp += s[cur];

        for (auto n: graph[cur])
            dfs(n);
    };

    for(int i=0; i<len; ++i)
    {
        if(seen.count(i))
            continue;

        idx.clear();
        tmp.clear();

        dfs(i);

        sort(tmp.begin(), tmp.end());
        sort(idx.begin(), idx.end());

        for (int k = 0; k < idx.size(); ++k)
            s[idx[k]] = tmp[k];
    }
    return s;
}

string Solution::smallestStringWithSwaps_unionFound(string s, vector<vector<int>>& pairs)
{
    int len = s.length();
    vector<int> p(len);
    for(int i=0; i<len; i++)
        p[i] = i;

    function<int(int)> find = [&](int x)
    {
        return p[x] == x ? x : find(p[x]);
    };

    // union
    for( auto& e: pairs)
    {
        p[find(e[0])] = find(e[1]);
    }

    for(int i=0; i<len; i++)
        cout << i << ": " << p[i] << "\n";

    vector<string> ss(len);
    vector<vector<int>> idx(len);
    for(int i=0; i<len; ++i)
    {
        int id = find(i);
        idx[id].push_back(i);
        ss[id].push_back(s[i]);
    }

    for(int i=0; i<len; ++i)
    {
        sort(ss[i].begin(), ss[i].end());
        for(int k=0; k<idx[i].size(); ++k)
            s[idx[i][k]] = ss[i][k];
    }
    return s;
}

int main()
{
    // s = "dcab", pairs = [[0,3],[1,2],[0,2]]
    string s = "dcab";
    vector<vector<int>> pairs;
    pairs.push_back({0, 3});
    pairs.push_back({1, 2});
    pairs.push_back({0, 2});

    Solution ss;
    cout << ss.smallestStringWithSwaps_unionFound(s, pairs) << "\n";
}

