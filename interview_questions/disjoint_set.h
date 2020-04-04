#pragma once

#include <numeric>
#include <vector>

class DisjointSet
{
public:
    DisjointSet(int n);
    int find(int x);
    bool unionFunc(int x, int y);

private:
    int find_recursive(int x);
    int find_iterative(int x);

private:
    std::vector<int> m_parent;
    std::vector<int> m_rank;
};
