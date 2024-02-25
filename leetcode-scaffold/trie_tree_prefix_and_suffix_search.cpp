#include "leetcode.h"
#include <string_view>

using namespace std;
using namespace osrm;

/*
leetcode: 745

Given many words, words[i] has weight i. Design a class WordFilter that supports one function, WordFilter.filter(String prefix, String suffix). 
It will return the word with given prefix and suffix with maximum weight. If no word exists, return -1.

Examples:
    Input:
    WordFilter(["apple"])
    WordFilter.filter("a", "e") // returns 0
    WordFilter.filter("b", "") // returns -1
Note: 
    * `words[i]` has length in range `[1, 10]`.
    * `prefix, suffix` have lengths in range `[0, 10]`.
    * `words[i]` and `prefix, suffix` queries consist of lowercase letters only.
*/

class WordFilter {
struct TrieNode {
    TrieNode() {
        val = -1;
        is_leaf = false;
        // assume that all inputs are consist of lowercase letters a-z.
        children.assign(26, NULL);
    }
    ~TrieNode() {
        for (auto n: children){
            delete n;
        }
    }
    int val;
    bool is_leaf;
    std::vector<TrieNode*> children;
};
public:
    void buildDict(const vector<string>& dict);
    int filter(const string& prefix, const string& suffix);
private:
    void insert(const string& key, int val);
private:
    TrieNode m_root;
};

void WordFilter::buildDict(const vector<string>& dict) {
    for (int i=0; i<dict.size(); i++) {
        insert(dict[i], i);
    }
}

void WordFilter::insert(const string& key, int val){
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

int WordFilter::filter(const string& prefix, const string& suffix) {
    TrieNode* p = &m_root;
    for (auto c: prefix) {
        if (p->children[c-'a'] == nullptr) {
            return -1;
        }
        p = p->children[c-'a'];
    }
    int ans = INT32_MIN;
    string buffer = prefix;
    function<void(TrieNode*)> backtrace = [&] (TrieNode* p) {
        if (p == nullptr) {
            return;
        }
        if (p->is_leaf) {
            if (buffer.rfind(suffix, buffer.size()-suffix.size()) != std::string::npos) {
                ans = std::max(ans, p->val);
            }
        }
        for (int i=0; i<26; ++i) {
            buffer.push_back(i+'a');
            backtrace(p->children[i]);
            buffer.pop_back();
        }
    };
    backtrace(p);
    return ans==INT32_MIN ? -1 : ans;
}

void WordFilter_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    WordFilter tm;
    int n = funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "buildDict") {
            tm.buildDict(funcArgs[i]);
        } else if (funcOperations[i] == "filter") {
            int actual = tm.filter(funcArgs[i][0], funcArgs[i][1]);
            if (actual != std::stoi(ans[i])) {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") failed. Expected: " << ans[i] << ", actual: " << actual;
            } else {
                util::Log(logINFO) << funcOperations[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") passed";
            }
        }
    }
}

int main() {
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
