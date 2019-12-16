#include "leetcode.h"

using namespace std;

class Solution
{
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts)
    {
        int len = accounts.size();
        vector<int> p(1001);
        for(int i=0; i<p.size(); ++i) p[i] = i;

        function<int(int)> find = [&](int x)
        {
            return p[x] == x ? x : find(p[x]);
        };

        function<void(int,int)> union_func = [&](int x, int y)
        {
            p[find(x)] = find(y);
        };

        int mailId = 0;
        unordered_map<string, int> email2NameId;
        unordered_map<string, int> email2MailId;
        for(int i=0; i<len; ++i)
        {
            for(int k=1; k<accounts[i].size(); ++k)
            {
                email2NameId[accounts[i][k]] = i;

                if(email2MailId.find(accounts[i][k]) == email2MailId.end())
                {
                    email2MailId[accounts[i][k]] = mailId++;
                }

                union_func(email2MailId[accounts[i][1]], email2MailId[accounts[i][k]]);
            }
        }

        unordered_map<int, vector<string>> tmp;
        for(auto& it: email2NameId)
        {
            int id = find(email2MailId[it.first]);
            tmp[id].push_back(it.first);
        }

        for(auto& it: tmp)
        {
            sort(it.second.begin(), it.second.end());
            it.second.erase(unique(it.second.begin(), it.second.end()), it.second.end());
        }

        int group = 0;
        vector<vector<string>> result(tmp.size());
        for(auto& it: tmp)
        {
            auto& name = accounts[email2NameId[it.second.front()]][0];
            result[group].push_back(name);
            result[group].insert(result[group].end(), it.second.begin(), it.second.end());
            ++group;
        }
        return result;
    }
};
