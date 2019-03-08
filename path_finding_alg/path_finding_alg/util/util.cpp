#include "util.h"

SquareGrid make_diagram1() 
{
	SquareGrid grid(30, 15);
	grid.addBlocks(3, 3, 5, 12);
	grid.addBlocks(13, 4, 15, 15);
	grid.addBlocks(21, 0, 23, 7);
	grid.addBlocks(23, 5, 26, 7);
	return grid;
}