#include "util.h"
#include "square_grid.h"
#include "grid_drawer.h"

#include "a_star_alg.h"
#include "dijkstra_alg.h"
#include "naive_bfs_alg.h"
#include "breadth_first_search.h"

void breadth_first_search_test_1()
{
	SimpleGraph exampleGraph { {
		{ 'A', { 'B' } },
		{ 'B', { 'A', 'C', 'D' } },
		{ 'C', { 'A' } },
		{ 'D', { 'A', 'E' } },
		{ 'E', { 'B' } }
	} };

	breadth_first_search(exampleGraph, 'A');
}

void breadth_first_search_test_2()
{
	SquareGrid grid = make_diagram1();
	GridLocation start {8, 7};
	GridLocation goal {17, 2};
	auto cameFrom = breadthFirstSearch(grid, start, goal);
	draw_grid(grid, 2, nullptr, &cameFrom);
}

void dijkstraSearch_test()
{
	WeightedGrid grid = make_diagram4();
	GridLocation start {1, 4};
	GridLocation goal {8, 5};
	std::unordered_map<GridLocation, GridLocation> cameFrom;
	std::unordered_map<GridLocation, double> costSofar;
	dijkstraSearch(grid, start, goal, cameFrom, costSofar);
	draw_grid(grid, 2, nullptr, &cameFrom);
	std::cout << '\n';
	draw_grid(grid, 3, &costSofar, nullptr);
	std::cout << '\n';
	std::vector<GridLocation> path = reconstructPath(start, goal, cameFrom);
	draw_grid(grid, 3, nullptr, nullptr, &path);
}

void a_star_search_test()
{
	WeightedGrid grid = make_diagram4();
	GridLocation start {1, 4};
	GridLocation goal {8, 5};
	std::unordered_map<GridLocation, GridLocation> cameFrom;
	std::unordered_map<GridLocation, double> costSofar;
	aStarSearch(grid, start, goal, cameFrom, costSofar);
	draw_grid(grid, 2, nullptr, &cameFrom);
	std::cout << '\n';
	draw_grid(grid, 3, &costSofar, nullptr);
	std::cout << '\n';
	std::vector<GridLocation> path = reconstructPath(start, goal, cameFrom);
	draw_grid(grid, 3, nullptr, nullptr, &path);
}

int main()
{
	// breadth_first_search_test_1();
	// breadth_first_search_test_2();

	// dijkstraSearch_test();

	a_star_search_test();
}
