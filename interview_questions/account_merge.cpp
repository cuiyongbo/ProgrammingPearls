#include "leetcode.h"

using namespace std;

class Solution
{
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);
};

vector<vector<string>> Solution::accountsMerge(vector<vector<string>>& accounts)
{
        const int MAX_EMAIL_COUNT = 1000*10+1;
        vector<int> p(MAX_EMAIL_COUNT);
        for(int i=0; i<MAX_EMAIL_COUNT; ++i) p[i] = i;

        function<int(int)> find = [&](int x)
        {
            if(p[x] != x)
            {
                p[x] = find(p[x]);
            }
            return p[x];
        };

        function<void(int,int)> union_func = [&](int x, int y)
        {
            p[find(x)] = find(y);
        };

        int accountId = 0, mailId = 0;
        unordered_map<string, int> email2MailId;
        unordered_map<string, int> email2AccountId;
        for(auto& account: accounts)
        {
            for(int k=1; k<account.size(); ++k)
            {
                email2AccountId[account[k]] = accountId;
                if(email2MailId.find(account[k]) == email2MailId.end())
                {
                    email2MailId[account[k]] = mailId++;
                }
                union_func(email2MailId[account[1]], email2MailId[account[k]]);
            }
            accountId++;
        }

        unordered_map<int, vector<string>> tmp;
        for(auto& it: email2AccountId)
        {
            int id = find(email2MailId[it.first]);
            tmp[id].push_back(it.first);
        }

        int group = 0;
        vector<vector<string>> result(tmp.size());
        for(auto& it: tmp)
        {
            auto& name = accounts[email2AccountId[it.second[0]]][0];
            result[group].push_back(name);

            sort(it.second.begin(), it.second.end());
            result[group].insert(result[group].end(), it.second.begin(), it.second.end());

            ++group;
        }
        return result;
}

int main()
{
    vector<vector<string>> accounts {
        {"John","johnsmith@mail.com","john_newyork@mail.com"},
        {"John","johnsmith@mail.com","john00@mail.com"},
        {"Mary","mary@mail.com"},
        {"John","johnnybravo@mail.com"}
    };

    Solution ss;
    vector<vector<string>> groups = ss.accountsMerge(accounts);

    for(auto& group: groups)
    {
        cout << group[0] << ":\n";
        for(int k=1; k<group.size(); ++k)
            cout << "\t" << group[k] << "\n";
    }
}
