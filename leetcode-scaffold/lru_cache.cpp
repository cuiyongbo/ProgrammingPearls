#include <iostream>
#include <map>
#include <unordered_map>
#include <list>
#include <cassert>
#include <shared_mutex>

using namespace std;

/*
Leetcode 146:

Design a data structure that follows the constraints of a Least Recently Used (LRUCache) cache.

Implement the LRUCache class:

`LRUCache(int capacity)` Initialize the LRUCache cache with positive size capacity.

`int get(int key)` Return the value of the key if the key exists, otherwise return -1.

`void put(int key, int value)` Update the value of the key if the key exists. 
Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
the capacity from this operation, evict the least recently used key.

Follow up: Could you do get and put in O(1) time complexity?

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
    LRUCache lRUCache = new LRUCache(2);
    lRUCache.put(1, 1); // cache is {1=1}
    lRUCache.put(2, 2); // cache is {1=1, 2=2}
    lRUCache.get(1);    // return 1
    lRUCache.put(3, 3); // LRUCache key was 2, evicts key 2, cache is {1=1, 3=3}
    lRUCache.get(2);    // returns -1 (not found)
    lRUCache.put(4, 4); // LRUCache key was 1, evicts key 1, cache is {4=4, 3=3}
    lRUCache.get(1);    // return -1 (not found)
    lRUCache.get(3);    // return 3
    lRUCache.get(4);    // return 4
*/

namespace std_implementation {

class LRUCache {
public:
    LRUCache(int cap) {
        m_capacity = cap;
    }

    int get(int key) {
        if (m_node_map.count(key) == 0) {
            return -1;
        }
        int val = m_node_map[key]->second;
        put(key, val);
        return val;
    }

    void put(int key, int value) {
        if (m_node_map.count(key)) {
            m_nodes.erase(m_node_map[key]);
        } else {
            if (m_capacity == m_nodes.size()) {
                m_node_map.erase(m_nodes.back().first);
                m_nodes.pop_back();
            }
        }
        m_nodes.push_front(std::make_pair(key, value));
        m_node_map[key] = m_nodes.begin();
    }

    void display() {
        for (auto it: m_nodes) {
            printf("(%d,%d)", it.first, it.second);
        }
        printf("\n");
    }

private:
    unordered_map<int, list<pair<int, int>>::iterator> m_node_map;
    list<pair<int, int>> m_nodes;
    int m_capacity;
};
}

namespace thread_safe_implementation {
class LRUCache {
public:
    LRUCache(int cap) {
        m_capacity = cap;
    }

    int get(int key) {
        std::shared_lock lock(m_mutex);
        auto it = m_node_map.find(key);
        if (it == m_node_map.end()) {
            return -1;
        } else {
            m_nodes.splice(m_nodes.begin(), m_nodes, it->second);
            return it->second->second;
        }
    }

    void put(int key, int value) {
        std::unique_lock lock(m_mutex);
        auto it = m_node_map.find(key);
        if (it != m_node_map.end()) {
            m_nodes.erase(it->second);
            m_node_map.erase(it);
        }
        m_nodes.push_front(std::make_pair(key, value));
        m_node_map[key] = m_nodes.begin();

        do_inviction_if_need();
    }

    void do_inviction_if_need() {
        while (m_nodes.size() > m_capacity) {
            m_node_map.erase(m_nodes.back().first);
            m_nodes.pop_back();
        }
    }

    void display() {
        std::shared_lock lock(m_mutex);
        for (auto it: m_nodes) {
            printf("(%d,%d)", it.first, it.second);
        }
        printf("\n");
    }

private:
    unordered_map<int, list<pair<int, int>>::iterator> m_node_map;
    list<pair<int, int>> m_nodes;
    int m_capacity;
    std::shared_mutex m_mutex;
};
}

int main() {
    //using std_implementation::LRUCache;
    using thread_safe_implementation::LRUCache;
    LRUCache lru(3);
    int p = lru.get(1);
    assert(p == -1);
    lru.put(1, 10);
    lru.put(2, 20);
    p = lru.get(1);
    assert(p == 10);
    lru.display();
    lru.put(3, 30);
    lru.put(4, 40);
    lru.display();
    lru.put(2, 22);
    lru.display();
    p = lru.get(2);
    assert(p == 22);
}