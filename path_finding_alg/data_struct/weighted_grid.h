#pragma once

#include "square_grid.h"

class WeightedGrid: public SquareGrid
{
	std::unordered_set<GridLocation> m_forests;
public:
	WeightedGrid(int weight, int height): SquareGrid(weight, height) {}
	
	double cost(GridLocation src, GridLocation dest) const
	{
        (void)src;
		return m_forests.find(dest) != m_forests.end() ? 5 : 1;
	}

    void addTree(GridLocation loc)
    {
        m_forests.emplace(loc);
    }
};
