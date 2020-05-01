#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 745 */

/*
Given many words, words[i] has weight i.

Design a class WordFilter that supports one function, WordFilter.filter(String prefix, String suffix). 
It will return the word with given prefix and suffix with maximum weight. If no word exists, return -1.

Examples:

    Input:
    WordFilter(["apple"])
    WordFilter.filter("a", "e") // returns 0
    WordFilter.filter("b", "") // returns -1

Note: words[i] and prefix, suffix queries consist of lowercase letters only.
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
            delete n;
    }

    int val;
    bool is_leaf;
    std::vector<TrieNode*> children;
};

class WordFilter 
{
public:
    void buildDict(const vector<string>& dict);
    int filter(const string& prefix, const string& suffix);

private:
    void insert(const string& key, int val);

private:
    TrieNode m_root;
};

void WordFilter::buildDict(const vector<string>& dict)
{
    for(int i=0; i<(int)dict.size(); i++)
        insert(dict[i], i);
}

void WordFilter::insert(const string& key, int val)
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

int WordFilter::filter(const string& prefix, const string& suffix)
{
    TrieNode* cur = &m_root;
    for(const auto& c: prefix)
    {
        cur = cur->children[c-'a'];
        if(cur == NULL) break;
    }

    if(cur == NULL) return -1;

    size_t suffix_len = suffix.length();

    int ans = -1;
    string courier;
    function<void(TrieNode*)> dfs = [&](TrieNode* cur)
    {
        if(cur == NULL) return;

        if(cur->is_leaf)
        {
            size_t size = courier.size();
            if(size >= suffix_len)
            {
                if(courier.substr(size-suffix_len) == suffix)
                    ans = std::max(cur->val, ans);
            }
        }

        for(int i=0; i<26; ++i)
        {
            if(cur->children[i] != NULL)
            {
                courier.push_back(i+'a');
                dfs(cur->children[i]);
                courier.pop_back();
            }
        }
    };

    dfs(cur);
    return ans;
}

void WordFilter_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    WordFilter tm;
    int n = (int)funcOperations.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "buildDict")
        {
            tm.buildDict(funcArgs[i]);
        }
        else if(funcOperations[i] == "filter")
        {
            int actual = tm.filter(funcArgs[i][0], funcArgs[i][1]);
            if(actual != std::stoi(ans[i]))
            {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << funcOperations[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") passed";
            }
        }
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();
    
    util::Log(logESSENTIAL) << "Running WordFilter tests:";
    TIMER_START(WordFilter);
    WordFilter_scaffold(
        "[WordFilter,buildDict,filter,filter,filter]", 
        "[[],[apple,abyss],[a,e],[b,],[a,]]",
        "[null,null,0,-1,1]");
    TIMER_STOP(WordFilter);
    util::Log(logESSENTIAL) << "WordFilter using " << TIMER_MSEC(WordFilter) << " milliseconds";
}
