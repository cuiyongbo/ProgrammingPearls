#include "leetcode.h"

using namespace std;

class Solution {
public:
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
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
};
