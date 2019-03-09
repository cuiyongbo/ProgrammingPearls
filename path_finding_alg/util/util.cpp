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

WeightedGrid make_diagram4() 
{
    WeightedGrid grid(10, 10);
    grid.addBlocks(1, 7, 4, 9);
    
    typedef GridLocation L;
    std::unordered_set<GridLocation> forests {
      L{3, 4}, L{3, 5}, L{4, 1}, L{4, 2},
      L{4, 3}, L{4, 4}, L{4, 5}, L{4, 6},
      L{4, 7}, L{4, 8}, L{5, 1}, L{5, 2},
      L{5, 3}, L{5, 4}, L{5, 5}, L{5, 6},
      L{5, 7}, L{5, 8}, L{6, 2}, L{6, 3},
      L{6, 4}, L{6, 5}, L{6, 6}, L{6, 7},
      L{7, 3}, L{7, 4}, L{7, 5}
    };
    for (auto& tree: forests)
    {
        grid.addTree(tree);
    }
    return grid;
}
