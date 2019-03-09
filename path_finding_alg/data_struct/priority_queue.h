#pragma once

#include <queue>
#include <vector>
#include <utility>

template <typename T, typename priority_t>
class PriorityQueue
{
    typedef std::pair<priority_t, T> PQElement;
    std::priority_queue<PQElement, std::vector<PQElement>, 
        std::greater<PQElement> > m_elements;
public:
    bool empty() const { return m_elements.empty(); }

    void put(T item, priority_t priority) 
    { 
        m_elements.emplace(priority, item); 
    }

    T get()
    {
        T item = m_elements.top().second;
        m_elements.pop();
        return item;
    }
};