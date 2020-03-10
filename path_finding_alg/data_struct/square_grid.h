#pragma once

#include <array>
#include <unordered_set>
#include "simple_graph_struct.h"

class SquareGrid
{
	int m_width, m_height;
	std::unordered_set<GridLocation> m_walls;
	static std::array<GridLocation, 4> m_dirs;

public:
	SquareGrid(int width, int height) : m_width(width), m_height(height)
	{
		m_walls.clear();
	}

	int width() const { return m_width; }
	int height() const { return m_height; }
	std::vector<GridLocation> neighbors(GridLocation id) const;

	bool inBounds(GridLocation id) const;
	bool passable(GridLocation id) const;
	void addBlocks(int xMin, int yMin, int xMax, int yMax);
};
