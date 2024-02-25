#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* 
leetcode: 211
Design a data structure that supports the following two operations:
    void addWord(word); // You may assume that word consists of lowercase letters a-z.
    bool search(word);
search(word) can search a literal word or a regular expression string containing only letters a-z or '.'. A '.' means it can represent any one letter.

Example:
    addWord("bad")
    addWord("dad")
    addWord("mad")
    search("pad") -> false
    search("bad") -> true
    search(".ad") -> true
    search("b..") -> true
*/

class WordDictionary {
public:
    void addWord(const string& word);
    bool search(const string& word);
private:
    TrieTree m_tree;
};

void WordDictionary::addWord(const string& word) {
    m_tree.insert(word);
}
    
/** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
bool WordDictionary::search(const string& word) {
    function<bool(TrieNode*, int)> backtrace = [&] (TrieNode* node, int u) {
        if (node == nullptr) {
            return false;
        }
        if (u == word.size()) {
            return node->is_leaf;
        }
        if (word[u] == '.') {
            for (auto c: node->children) {
                if (backtrace(c, u+1)) {
                    return true;
                }
            }
            return false;
        } else {
            return backtrace(node->children[word[u]-'a'], u+1);
        }
    };
    return backtrace(m_tree.root(), 0);
}

void WordFilter_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    WordDictionary tm;
    int n = (int)funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "addWord") {
            tm.addWord(funcArgs[i][0]);
        } else if(funcOperations[i] == "search") {
            bool actual = tm.search(funcArgs[i][0]);
            string actual_str = actual ? "true" : "false";
            if (actual_str != ans[i]) {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed, Expected: " << ans[i] << ", actual: " << actual_str;
            } else {
                util::Log(logINFO) << funcOperations[i] << "(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    
    util::Log(logESSENTIAL) << "Running WordDictionary tests:";
    TIMER_START(WordDictionary);
    WordFilter_scaffold(
        "[WordDictionary,addWord,addWord,addWord,search,search,search,search,search]", 
        "[[],[bad],[dad],[mad],[pad],[bad],[.ad],[b..],[...]]",
        "[null,null,null,null,false,true,true,true,true]");
    TIMER_STOP(WordDictionary);
    util::Log(logESSENTIAL) << "WordDictionary using " << TIMER_MSEC(WordDictionary) << " milliseconds";
}
