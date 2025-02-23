#include "leetcode.h"

using namespace std;

/*
leetcode: 460

Design and implement a data structure for Least Frequently Used (LFU) cache. 
It should support the following operations: get and put.

`get(key)` - Get the value (will always be positive) of the key 
if the key exists in the cache, otherwise return -1.

`put(key, value)` - Set or insert the value if the key is not already present. 
When the cache reaches its capacity, it should invalidate the least frequently 
used item before inserting a new item. For the purpose of this problem, 
when there is a tie (i.e., two or more keys that have the same frequency),
the least recently used key would be evicted.

Note that the number of times an item is used is the number of calls to 
the get and put functions for that item since it was inserted. 
This number is set to zero when the item is removed.

Follow up: Could you do both operations in O(1) time complexity?

Example:
    LFUCache cache = new LFUCache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // returns 1
    cache.put(3, 3);    // evicts key 2
    cache.get(2);       // returns -1 (not found)
    cache.get(3);       // returns 3.
    cache.put(4, 4);    // evicts key 1.
    cache.get(1);       // returns -1 (not found)
    cache.get(3);       // returns 3
    cache.get(4);       // returns 4

Hint: use insertSort to maintain order: first sort by frequency then by timestamp
Reference solution: https://zxi.mytechroad.com/blog/hashtable/leetcode-460-lfu-cache/
*/

namespace naive_implementation {
class LFUCache {
private:
    struct CacheNode {
        int key;
        int val;
        int access;
        time_t timestamp;
        CacheNode* prev;
        CacheNode* next;
        CacheNode(int k=-1, int v=-1):
            key(k), val(v),
            prev(nullptr),
            next(nullptr) {
            access = 1;
            timestamp = time(nullptr);
        }
    };
    int m_capacity;
    CacheNode m_head;
    CacheNode m_tail;
    map<int, CacheNode*> m_nodeMap;    
public:
    LFUCache(int capacity):
        m_capacity(capacity) {
        assert(capacity > 0);
        m_head.next = &m_tail;
        m_tail.prev = &m_head;
    }
    
    int get(int key) {
        auto it = m_nodeMap.find(key);
        if (it == m_nodeMap.end()) {
            return -1;
        } else {
            auto node = it->second;
            node->access++;
            node->timestamp = time(nullptr);
            // extract node from the list
            node->prev->next = node->next; node->next->prev = node->prev;
            // re-insert node into the list
            insert(node);
            return node->val;
        }
    }
    
    void put(int key, int value) {
        CacheNode* node = nullptr;
        auto it = m_nodeMap.find(key);
        if (it == m_nodeMap.end()) {
            // node doesn't exist yet
            node = new CacheNode(key, value);
            // if we reach the capcity, we need to remove the last node from tail first
            if (m_nodeMap.size() == m_capacity) {
                auto p = m_tail.prev;
                p->prev->next = &m_tail;
                m_tail.prev = p->prev;
                m_nodeMap.erase(p->key);
                delete p;  
            }
            m_nodeMap[key] = node;
        } else {
            // node has already existed, update its value
            node = it->second;
            node->val = value;
            node->access++;
            node->timestamp = time(nullptr);
            // extract node from the list
            node->prev->next = node->next; node->next->prev = node->prev;
        }
        // re-insert node into the list
        insert(node);
    }
    void display() {
        printf("Capacity: %d, Value: ", m_capacity);
        auto p = m_head.next;
        while (p != &m_tail) {
            printf("(%d,%d,%d,%lld)", p->key, p->val, p->access, (int64_t)p->timestamp);
            p = p->next;
        }
        printf("\n");
    }
private:
    void insert(CacheNode* node) {
        // perform insertionSort to find where to insert node
        auto p = m_head.next;
        while (p != &m_tail) {
            if (p->access > node->access) {
                p = p->next;
            } else if (p->access == node->access) {
                if (p->timestamp > node->timestamp) {
                    p = p->next;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        // insert node before p
        // point node's previous to [p->prev]
        node->prev = p->prev;
        // point node's next to [p]
        node->next = p;
        // point [p->prev]'s next to node
        p->prev->next = node;
        // point p's previous to node
        p->prev = node; 
    }
};

}

uint64_t get_timetick_count() {
    auto tick_count = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tick_count.time_since_epoch()).count();
}

class LFUCache {
private:
    struct CacheNode {
        int key;
        int val;
        int access;
        uint64_t timestamp;
        CacheNode(int k=0, int v=0) {
            key = k;
            val = v;
            access = 1;
            timestamp = get_timetick_count();
        }
    };
    int m_capcity;
    std::list<CacheNode> m_nodes;
    std::map<int, std::list<CacheNode>::iterator> m_node_map;

public:

    LFUCache(int cap) {
        m_capcity = cap;
    }

    int get(int key) {
        auto it = m_node_map.find(key);
        if (it == m_node_map.end()) {
            return -1;
        } else {
            auto node = *(it->second);
            // update its value
            node.access++;
            node.timestamp = get_timetick_count();
            // remove old reference
            m_nodes.erase(it->second);
            m_node_map.erase(it);
            insert(node);
            return node.val;
        }
    }

    void put(int key, int val) {
        auto it = m_node_map.find(key);
        if (it == m_node_map.end()) {
            // key doesn't exist yet
            // remove the last node if we reach the capacity
            if (m_nodes.size() == m_capcity) {
                auto b = m_nodes.back();
                m_node_map.erase(b.key);
                m_nodes.pop_back();
            }
            // insert node
            CacheNode node(key, val);
            insert(node);
        } else {
            // key has already existed
            CacheNode node = *(it->second);
            // update its value
            node.val = val;
            node.access++;
            node.timestamp = get_timetick_count();
            // remove old reference
            m_nodes.erase(it->second);
            m_node_map.erase(it);
            // re-insert node
            insert(node);
        }
    }

    void insert(const CacheNode& node) {
        int index = 0;
        for (const auto& it: m_nodes) {
            if (node.access > it.access) {
                break;
            } else if (node.access == it.access) {
                if (node.timestamp > it.timestamp) {
                    break;
                }
            }
            index++;
        }
        m_nodes.insert(std::next(m_nodes.begin(), index), node);
        m_node_map[node.key] = std::next(m_nodes.begin(), index); // IMPORTANT: you have to re-fetch iterator since it has changed to the new node after `m_nodes.insert`
    }

    void display() {
        printf("Capacity: %d, Value: ", m_capcity);
        for (const auto& n : m_nodes) {
            printf("(%d,%d,%d,%lld)", n.key, n.val, n.access, (int64_t)n.timestamp);
        }
        printf("\n");
    }
};


int main () {
    LFUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cache.display(); // (2,2,1,760850823724625)(1,1,1,760850823724208)
    int p = cache.get(1);
    assert(p == 1);
    cache.display(); // (1,1,2,760850823776000)(2,2,1,760850823724625)
    cache.put(3, 3); // evicts key 2
    cache.display(); // (1,1,2,760850823776000)(3,3,1,760850823789791)
    p = cache.get(2);
    assert(p == -1);
    p = cache.get(3);
    assert(p == 3);
    cache.put(4, 4); // evicts key 1.
    cache.display(); // (3,3,2,760850823801166)(4,4,1,760850823802000)
    p = cache.get(2);
    assert(p == -1);
    p = cache.get(3);
    assert(p == 3);
    p = cache.get(4);
    assert(p == 4);
    cache.display(); // (3,3,3,760850823812500)(4,4,2,760850823813208)
    cout << p << endl;
}
