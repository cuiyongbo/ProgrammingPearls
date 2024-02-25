#pragma once

#include <numeric>
#include <vector>

class DisjointSet {
public:
    DisjointSet(int n);

    // return the group id to which x belongs
    int find(int x);

    // return false if x and y belong to the same component, otherwise true
    bool unionFunc(int x, int y);

private:
    int find_recursive(int x);
    int find_iterative(int x);

private:
    std::vector<int> m_parent;
    std::vector<int> m_rank; // number of edges in the longest simple path between node x and one of its descendent leaves
};
