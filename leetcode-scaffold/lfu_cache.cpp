#include <iostream>
#include <map>
#include <time.h>
#include <assert.h>

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
            node->prev->next = node->next;
            node->next->prev = node->prev;

            insert(node);
            return node->val;
        }
    }
    
    void put(int key, int value) {
        CacheNode* node = nullptr;
        auto it = m_nodeMap.find(key);
        if (it == m_nodeMap.end()) {
            node = new CacheNode(key, value);
            if (m_nodeMap.size() == m_capacity) {
                auto p = m_tail.prev;
                p->prev->next = &m_tail;
                m_tail.prev = p->prev;
                m_nodeMap.erase(p->key);
                delete p;  
            }
            m_nodeMap[key] = node;
        } else {
            node = it->second;
            node->val = value;
            node->access++;
            node->timestamp = time(nullptr);
            node->prev->next = node->next;
            node->next->prev = node->prev;
        }
        insert(node);
    }
    void display() {
        cout << "Capacity: " << m_capacity << endl;
        cout << "Value: ";
        auto p = m_head.next;
        while (p != &m_tail) {
            cout << "(" << p->key << ", " << p->val << "," << p->access << "," << p->timestamp << ")";
            p = p->next;
        }
        cout << endl;
    }
private:
    void insert(CacheNode* node) {
        // perform insertionSort to find insert position
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
        node->next = p;
        node->prev = p->prev;
        p->prev->next = node;
        p->prev = node; 
    }
};

int main () {
    LFUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cache.display();
    cout << cache.get(1) << endl;       // returns 1
    cache.display();
    cache.put(3, 3);    // evicts key 2
    cout << cache.get(2) << endl;       // returns -1 (not found)
    cout << cache.get(3) << endl;       // returns 3.
    cache.put(4, 4);    // evicts key 1.
    cout << cache.get(1) << endl;       // returns -1 (not found)
    cout << cache.get(3) << endl;       // returns 3
    cout << cache.get(4) << endl;       // returns 4
}
