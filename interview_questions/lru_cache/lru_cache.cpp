#include <iostream>
#include <list>
#include <map>
#include <string>
#include <cassert>

using namespace std;

class LruCache
{
public:
    LruCache(int n): m_cacheCapacity(n) 
    {
        m_cacheList.head.next = &(m_cacheList.tail);
        m_cacheList.tail.prev = &(m_cacheList.head);
    }

    ~LruCache();

    string get(string key);
    void set(string key, string value);
    void display();

private:
    int m_cacheCapacity;

    struct CacheNode
    {
        string key;
        string value;

        CacheNode* next;
        CacheNode* prev;

        CacheNode() : next(NULL), prev(NULL) {}

        ~CacheNode()
        {
            // delete next;
            // delete prev;
        }
    };

    struct CacheList
    {
        CacheNode head;
        CacheNode tail;
    };

    CacheList m_cacheList;
    map<string, CacheNode*> m_cacheMap;
};

string LruCache::get(string key)
{
    auto it = m_cacheMap.find(key);
    if(it != m_cacheMap.end())
    {
        auto node = m_cacheMap[key];

        // remove node
        node->prev->next = node->next;
        node->next->prev = node->prev;

        // move node to front
        node->next = m_cacheList.head.next;
        m_cacheList.head.next->prev = node;
        node->prev = &(m_cacheList.head);
        m_cacheList.head.next = node;

        return node->value;
    }
    else
    {
        return "";
    }
}

void LruCache::set(string key, string value)
{
    auto it = m_cacheMap.find(key);
    if(it == m_cacheMap.end())
    {
        if(m_cacheMap.size() == m_cacheCapacity)
        {
            auto oldest = m_cacheList.tail.prev;
            oldest->prev->next = oldest->next;
            oldest->next->prev = oldest->prev;

            oldest->next = NULL;
            oldest->prev = NULL;

            m_cacheMap.erase(oldest->key);
            delete oldest;
        }

        // insert node to front
        auto node = new CacheNode;
        node->key = key;
        node->value = value;
        node->next = m_cacheList.head.next;
        m_cacheList.head.next->prev = node;
        node->prev = &(m_cacheList.head);
        m_cacheList.head.next = node;

        m_cacheMap[key] = node;
    }
    else
    {
        auto node = m_cacheMap[key];
        node->value = value;

        // remove node
        node->prev->next = node->next;
        node->next->prev = node->prev;

        // move node to front
        node->next = m_cacheList.head.next;
        m_cacheList.head.next->prev = node;
        node->prev = &(m_cacheList.head);
        m_cacheList.head.next = node;
    }
}

void LruCache::display()
{
    cout << "Cache: " << endl;
    for(auto p = m_cacheList.head.next; p != &(m_cacheList.tail); p = p->next)
    {
        cout << p->key << ": " << p->value << endl;
    }
}

LruCache::~LruCache()
{
    for(auto p = m_cacheList.head.next; p != &(m_cacheList.tail); )
    {
        auto q = p->next;
        delete p;
        p = q;
    }
}



int main()
{
    LruCache lru(3);

    lru.set("name", "cyb");
    lru.set("hello", "world");
    lru.set("foo", "bar");
    lru.set("memcached", "redis");
    
    lru.display();

    assert(lru.get("hello") == "world");
    
    lru.display();
    
    assert(lru.get("foo") == "bar");

    lru.display();

    lru.set("name", "cyb");
    assert(lru.get("memcached") == "");

    lru.display();
}
