#pragma once

#include <vector>
#include <numeric>
#include <unordered_set>
#include <stack>

class DSU
{
public:
    DSU(int count)
    {
        m_aux.resize(count);
		std::iota(m_aux.begin(), m_aux.end(), 0);
    }

    int find(int x)
    {
        if(m_aux[x] != x)
        {
            m_aux[x] = find(m_aux[x]);
        }
        return m_aux[x];
    }

    void unionFunc(int x, int y)
    {
        m_aux[find(x)] = find(y);
    }

    int groupCount()
    {
    	std::unordered_set<int> groups;
    	for(int i=0; i<m_aux.size(); ++i)
    	{
    		groups.emplace(find(i));
    	}
    	return groups.size();
    }

private:
    std::vector<int> m_aux;
};

class DisjointSet
{
public:
    DisjointSet(int n)
    {
        m_rank.resize(n+1, 0);
        m_parent.resize(n+1, 0);
        std::iota(m_parent.begin(), m_parent.end(), 0);
    }

    int find(int x)
    {
        std::stack<int> s;
        while(x != m_parent[x])
        {
            s.push(x);
            x = m_parent[x];
        }

        while(!s.empty())
        {
            m_parent[s.top()] = x;
            s.pop();
        }
        return m_parent[x];
    }

    bool unionFunc(int x, int y)
    {
        int px = find(x);
        int py = find(y);

        if(px == py) return false; // cycle or multi-pointing detected

        if(m_rank[px] > m_rank[py])
        {
            m_parent[py] = px;
        }
        else
        {
            m_parent[px] = py;
            if(m_rank[px] == m_rank[py])
                ++m_rank[py];
        }
        return true;
    }

private:
    std::vector<int> m_parent;
    std::vector<int> m_rank;
};