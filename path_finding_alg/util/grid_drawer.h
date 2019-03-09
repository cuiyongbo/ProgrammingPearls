#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iterator>

template<typename Location>
std::vector<Location> reconstructPath(Location start, Location goal,
    std::unordered_map<Location, Location>& cameFrom)
{
    std::vector<Location> path;
    Location current = goal;
    while(current != start)
    {
        path.push_back(current);
        current = cameFrom[current];
    }
    path.push_back(start);
    std::reverse(path.begin(),path.end());
    return path;
}

// This outputs a grid. Pass in a `distances` map if you want to print
// the distances, or pass in a `point_to` map if you want to print
// arrows that point to the parent location, or pass in a `path` vector
// if you want to draw the path.
template<class Graph>
void draw_grid(const Graph& graph, int field_width,
	std::unordered_map<GridLocation, double>* distances = nullptr,
	std::unordered_map<GridLocation, GridLocation>* point_to = nullptr,
	std::vector<GridLocation>* path = nullptr)
{
	for (int y = 0; y != graph.height(); ++y)
	{
		for (int x = 0; x != graph.width(); ++x)
		{
			GridLocation id{ x, y };
			std::cout << std::left << std::setw(field_width);
			if (!graph.passable(id))
			{
				std::cout << std::string(field_width, '#');
			}
			else if (point_to != nullptr && point_to->count(id))
			{
				GridLocation next = (*point_to)[id];
				if (next.x == x + 1) { std::cout << "> "; }
				else if (next.x == x - 1) { std::cout << "< "; }
				else if (next.y == y + 1) { std::cout << "v "; }
				else if (next.y == y - 1) { std::cout << "^ "; }
				else { std::cout << "* "; }
			}
			else if (distances != nullptr && distances->count(id))
			{
				std::cout << (*distances)[id];
			}
			else if (path != nullptr && std::find(path->begin(), path->end(), id) != path->end())
			{
				if(id == *(path->begin()))
				{
					std::cout << 'S';
				}
				else if(id == *(path->rbegin()))
				{
					std::cout << 'E';
				}
				else
				{
					std::cout << '@';
				}
			}
			else
			{
				std::cout << '.';
			}
		}
		std::cout << '\n';
	}
}