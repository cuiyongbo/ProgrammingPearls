#pragma once

#include <iostream>
#include <unordered_map>
#include <priority_queue.h>
#include "simple_graph_struct.h"

inline double heuristic(GridLocation l, GridLocation r)
{
    return abs(l.x - l.x) + abs(l.y - r.y);
}

template <typename Location, typename Graph>
void aStarSearch(Graph& graph, Location start, Location goal,
    std::unordered_map<Location, Location>& cameFrom,
    std::unordered_map<Location, double>& costSofar)
{
    PriorityQueue<Location, double> frontier;
    frontier.put(start, 0);

    cameFrom[start] = start;
    costSofar[start] = 0;
    while(!frontier.empty())
    {
        Location current = frontier.get();
        if (current == goal)
            break;

        for (Location& next: graph.neighbors(current))
        {
            double newCost = costSofar[current] + graph.cost(current, next);
            if (costSofar.find(next) == costSofar.end()
                || newCost < costSofar[next]) {
                costSofar[next] = newCost;
                cameFrom[next] = current;
                frontier.put(next, newCost + heuristic(next, goal));
            }
        }
    }
}

template<typename Graph>
void aStarSearch(Graph& graph,
    typename Graph::Location start,
    typename Graph::Location goal,
    std::function<typename Graph::cost_t (typename Graph::Location, typename Graph::Location)> heuristic,
    std::unordered_map<typename Graph::Location, typename Graph::Location>& cameFrom,
    std::unordered_map<typename Graph::Location, typename Graph::cost_t>& costSofar)
{
    typedef typename Graph::Location Location;
    typedef typename Graph::cost_t cost_t;
    PriorityQueue<Location, cost_t> frontier;
    frontier.put(start, 0);

    cameFrom[start] = start;
    costSofar[start] = cost_t(0);

    std::vector<Location> neighbors;
    while(!frontier.empty())
    {
        Location current = frontier.get();
        if (current == goal)
            break;

        graph.get_neighbors(current, neighbors);
        for (auto& next : neighbors)
        {
            cost_t newCost = costSofar[current] + graph.cost(current, next);
            if (costSofar.find(next) == costSofar.end()
                || newCost < costSofar[next])
            {
                costSofar[next] = newCost;
                cameFrom[next] = current;
                frontier.put(next, newCost+heuristic(next, goal));
            }
        }
    }
}














