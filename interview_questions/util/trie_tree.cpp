#include "trie_tree.h"

using namespace std;

void TrieTree::insert(const string& word)
{
    TrieNode* current = m_root.get();
    for(const auto& c: word)
    {
        if(current->children[c-'a'] == NULL)
        {
            current->children[c-'a'] = new TrieNode;
        }
        current = current->children[c-'a'];
    }
    current->is_leaf = true;
}

bool TrieTree::search(const string& word) const
{
    TrieNode* current = find(word);
    return current != NULL && current->is_leaf;
}

bool TrieTree::startsWith(const string& word) const
{
    return find(word) != NULL;
}

TrieNode* TrieTree::find(const string& word) const
{
    TrieNode* current = m_root.get();
    for(const auto& c: word)
    {
        current = current->children[c-'a'];
        if(current == NULL) break;
    }
    return current;
}
