#pragma once

#include <queue>
#include <iostream>
#include <unordered_set>
#include "simple_graph_struct.h"

void breadth_first_search(SimpleGraph& graph, char start)
{
    std::queue<char> frontizer;
    frontizer.push(start);

    std::unordered_set<char> visited;
    visited.insert(start);

    while (!frontizer.empty())
    {
        char current = frontizer.front();
        std::cout << "Visiting " << current << '\n';
        frontizer.pop();
        for (auto next: graph.neighbors(current))
        {
            if (visited.find(next) == visited.end())
            {
                frontizer.push(next);
                visited.insert(next);
            }
        }
    }
}