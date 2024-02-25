#include "leetcode.h"

using namespace std;
using namespace osrm;

/*
leetcode: 677 

Implement a MapSum class with insert, and sum methods.

For the method insert, you'll be given a pair of (string, integer). The string represents the key and the integer represents the value.
If the key already existed, then the original key-value pair will be overridden to the new one.
For the method sum, you'll be given a string representing the prefix, and you need to return the sum of all the pairs' value whose key starts with the prefix.
you may assume that key and prefix consist of only lowercase English letters.

Example 1:
    Input: insert("apple", 3), Output: Null
    Input: sum("ap"), Output: 3
    Input: insert("app", 2), Output: Null
    Input: sum("ap"), Output: 5
*/

struct TrieNode {
    TrieNode() {
        val = 0;
        is_leaf = false;
        // assume that all inputs are consist of lowercase letters a-z.
        children.assign(26, nullptr);
    }

    ~TrieNode() {
        for (auto n: children) {
            delete n;
        }
    }

    int val;
    bool is_leaf;
    std::vector<TrieNode*> children;
};

class MapSum {
public:
    void insert(const string& key, int val);
    int sum(const string& prefix);
private:
    TrieNode m_root;
};

void MapSum::insert(const string& key, int val) {
    TrieNode* cur = &m_root;
    for (auto c: key) {
        if (cur->children[c-'a'] == nullptr) {
            cur->children[c-'a'] = new TrieNode;
        }
        cur = cur->children[c-'a'];
    }
    cur->val = val;
    cur->is_leaf = true;
}

int MapSum::sum(const string& prefix) {
    TrieNode* cur = &m_root;
    for (auto c: prefix) {
        cur = cur->children[c-'a'];
        if (cur == nullptr) {
            break;
        }
    }
    function<int(TrieNode*)> dfs = [&] (TrieNode* node) {
        if (node == nullptr) {
            return 0;
        }
        int ans = 0;
        if (node->is_leaf) {
            ans += node->val;
        }
        for (auto c: node->children) {
            ans += dfs(c);
        }
        return ans;
    };
    return dfs(cur);
}

void MapSum_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MapSum tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "insert") {
            tm.insert(funcArgs[i][0], std::stoi(funcArgs[i][1]));
        } else if (funcOperations[i] == "sum") {
            int actual = tm.sum(funcArgs[i][0]);
            if (actual != std::stoi(ans[i])) {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed. Expected: " << ans[i] << ", actual: " << actual;
            } else {
                util::Log(logINFO) << funcOperations[i] << "(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    
    util::Log(logESSENTIAL) << "Running MapSum tests:";
    TIMER_START(MapSum);
    MapSum_scaffold(
        "[MapSum,insert,sum,insert,sum,insert,sum,sum]",
        "[[],[apple,3],[ap],[apm,2],[ap],[approve,7],[ap],[app]]",
        "[null,null,3,null,5,null,12,10]");
    TIMER_STOP(MapSum);
    util::Log(logESSENTIAL) << "MapSum using " << TIMER_MSEC(MapSum) << " milliseconds";
}
