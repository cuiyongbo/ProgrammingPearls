#include "a_star.h"
#include <iostream>
#include <queue>
#include <unordered_set>

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
