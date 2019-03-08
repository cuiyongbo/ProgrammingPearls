#include "util.h"
#include "grid_drawer.h"
#include "a_star.h"
#include "square_grid.h"

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

int main()
{
	// breadth_first_search_test_1();

	SquareGrid grid = make_diagram1();
	draw_grid(grid, 2);

}
