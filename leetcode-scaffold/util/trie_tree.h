#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
struct TrieNode {
    TrieNode() {
        is_leaf = false;
        // assume that all inputs ONLY consist of ASCII characters.
        children.assign(128, nullptr);
    }
    ~TrieNode() {
        for (auto n: children) {
            delete n;
        }
    }
    bool is_leaf;
    std::vector<TrieNode*> children;
};

class TrieTree {
public:
    TrieTree() {
        m_root = std::unique_ptr<TrieNode>(new TrieNode);
    }

    void insert(const std::string& word);
    bool search(const std::string& word) const;
    bool startsWith(const std::string& prefix) const;

    TrieNode* root() const { 
        return m_root.get();
    }

private:
    TrieNode* find(const std::string& word) const;

private:
    std::unique_ptr<TrieNode> m_root;
};
