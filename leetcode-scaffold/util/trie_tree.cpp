#include "trie_tree.h"

using namespace std;

void TrieTree::insert(const string& word) {
    TrieNode* cur = m_root.get();
    for (const auto& c: word) {
        if (cur->children[c] == nullptr) {
            cur->children[c] = new TrieNode;
        }
        cur = cur->children[c];
    }
    cur->is_leaf = true;
}

bool TrieTree::search(const string& word) const {
    TrieNode* cur = find(word);
    return cur != nullptr && cur->is_leaf;
}

bool TrieTree::startsWith(const string& word) const {
    return find(word) != nullptr;
}

TrieNode* TrieTree::find(const string& word) const {
    TrieNode* cur = m_root.get();
    for (const auto& c: word) {
        cur = cur->children[c];
        if (cur == nullptr) {
            break;
        }
    }
    return cur;
}
