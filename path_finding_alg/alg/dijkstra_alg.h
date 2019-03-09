#include "priority_queue.h"
#include <unordered_map>

template <typename Location, typename Graph>
void dijkstraSearch(Graph& graph, Location start, Location goal,
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
                frontier.put(next, newCost);
            }
        }
    }
}
