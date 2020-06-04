#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 677 */

/*
Implement a MapSum class with insert, and sum methods.

For the method insert, you’ll be given a pair of (string, integer). 
The string represents the key and the integer represents the value. 
If the key already existed, then the original key-value pair will 
be overridden to the new one.

For the method sum, you’ll be given a string representing the prefix, 
and you need to return the sum of all the pairs’ value whose key starts with the prefix.

Example 1:

    Input: insert("apple", 3), Output: Null
    Input: sum("ap"), Output: 3
    Input: insert("app", 2), Output: Null
    Input: sum("ap"), Output: 5
*/

struct TrieNode
{
    TrieNode()
    {
        val = 0;
        is_leaf = false;

        // assume that all inputs are consist of lowercase letters a-z.
        children.assign(26, NULL);
    }

    ~TrieNode()
    {
        for(auto n: children) 
        {
#ifdef DEBUG
            std::cout << "delete: " << n << std::endl;
#endif
            delete n;
        }
    }

    int val;
    bool is_leaf;
    std::vector<TrieNode*> children;
};

class MapSum 
{
public:
    void insert(const string& key, int val);
    int sum(const string& prefix);

private:
    TrieNode m_root;
};

void MapSum::insert(const string& key, int val)
{
    TrieNode* cur = &m_root;
    for(const auto& c: key)
    {
        if(cur->children[c-'a'] == NULL)
            cur->children[c-'a'] = new TrieNode;
        cur = cur->children[c-'a'];
    }
    cur->is_leaf = true;
    cur->val = val;
}

int MapSum::sum(const string& prefix)
{
    TrieNode* cur = &m_root;
    for(const auto& c: prefix)
    {
        cur = cur->children[c-'a'];
        if(cur == NULL) break;
    }

    function<int(TrieNode*)> dfs = [&](TrieNode* cur)
    {
        int ans = 0;
        if(cur == NULL) return ans;
        if(cur->is_leaf) ans += cur->val;

        for(auto n: cur->children)
           ans += dfs(n);

        return ans;
    };

    return dfs(cur);
}


void MapSum_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MapSum tm;
    int n = (int)funcOperations.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "insert")
        {
            tm.insert(funcArgs[i][0], std::stoi(funcArgs[i][1]));
        }
        else if(funcOperations[i] == "sum")
        {
            int actual = tm.sum(funcArgs[i][0]);
            if(actual != std::stoi(ans[i]))
            {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << funcOperations[i] << "(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();
    
    util::Log(logESSENTIAL) << "Running MapSum tests:";
    TIMER_START(MapSum);
    MapSum_scaffold(
        "[MapSum,insert,sum,insert,sum]", 
        "[[],[apple,3],[ap],[app,2],[ap]]",
        "[null,null,3,null,5]");
    TIMER_STOP(MapSum);
    util::Log(logESSENTIAL) << "MapSum using " << TIMER_MSEC(MapSum) << " milliseconds";
}
