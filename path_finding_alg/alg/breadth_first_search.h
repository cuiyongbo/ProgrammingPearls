#pragma once

#include <queue>
#include <unordered_map>

template<typename Location, typename Graph>
std::unordered_map<Location, Location>
breadthFirstSearch(Graph graph, Location start, Location goal)
{
	std::queue<Location> frontier;
	frontier.push(start);

	std::unordered_map<Location, Location> cameFrom;
	cameFrom[start] = start;
	
	while(!frontier.empty())
	{
		Location current = frontier.front();
		frontier.pop();
	
		if(current == goal)
			break;

		for (Location& next : graph.neighbors(current))
		{
			if(cameFrom.find(next) == cameFrom.end())
			{
				frontier.push(next);
				cameFrom[next] = current;
			}
		}
	}

	return cameFrom;
}
