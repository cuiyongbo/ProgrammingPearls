#include <iostream>
#include <map>
#include <cassert>

using namespace std;

/*
Leetcode 146:

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

`LRUCache(int capacity)` Initialize the LRU cache with positive size capacity.

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
    lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    lRUCache.get(2);    // returns -1 (not found)
    lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    lRUCache.get(1);    // return -1 (not found)
    lRUCache.get(3);    // return 3
    lRUCache.get(4);    // return 4
*/

class LRUCache {
public:
    LRUCache(int capacity):
        m_capacity(capacity) {
        assert(capacity > 0);
        m_head.next = &m_tail;
        m_tail.prev = &m_head;
    }

    ~LRUCache() {
        m_nodeMap.clear();
        auto p = m_head.next;
        while (p != &m_tail) {
            auto q = p->next;
            delete p;
            p = q;
        }
    }
    
    int get(int key) {
        auto it = m_nodeMap.find(key);
        if (it == m_nodeMap.end()) {
            return -1;
        } else {
            auto node = it->second;
            node->prev->next = node->next;
            node->next->prev = node->prev; 
            node->prev = &m_head;
            node->next = m_head.next;
            m_head.next->prev = node;
            m_head.next = node; 
            return node->val;
        }
    }
    
    void put(int key, int value) {
        CacheNode* node = nullptr;
        auto it = m_nodeMap.find(key);
        if (it == m_nodeMap.end()) {
            node = new CacheNode(key, value);
            if (m_nodeMap.size() == m_capacity) {
                auto node_to_evict = m_tail.prev;
                node_to_evict->prev->next = node_to_evict->next;
                node_to_evict->next->prev = node_to_evict->prev;
                m_nodeMap.erase(node_to_evict->key);
                delete node_to_evict;
            }
            m_nodeMap[key] = node;
        } else {
            node = it->second;
            node->val = value;
            node->prev->next = node->next;
            node->next->prev = node->prev;    
        }

        // the execution order matters
        node->prev = &m_head;
        node->next = m_head.next;
        m_head.next->prev = node;
        m_head.next = node;
    }

    void display() {
        cout << "Capacity: " << m_capacity << endl;
        cout << "Value: ";
        auto p = m_head.next;
        while (p != &m_tail) {
            cout << "(" << p->key << ", " << p->val << ")";
            p = p->next;
        }
        cout << endl;
    }
private:
    struct CacheNode {
        int key;
        int val;
        CacheNode* prev;
        CacheNode* next;
        CacheNode(int k=-1, int v=-1):
            key(k), val(v),
            prev(nullptr),
            next(nullptr) {
        }
    };
    int m_capacity;
    CacheNode m_head;
    CacheNode m_tail;
    map<int, CacheNode*> m_nodeMap;
};

int main() {
    LRUCache* lru = new LRUCache(3);
    int p = lru->get(1);
    assert(p == -1);
    lru->put(1, 10);
    lru->put(2, 20);
    p = lru->get(1);
    assert(p == 10);
    lru->display();
    lru->put(3, 30);
    lru->put(4, 40);
    lru->display();
    lru->put(2, 22);
    lru->display();
    p = lru->get(2);
    assert(p == 22);
    delete lru;
}