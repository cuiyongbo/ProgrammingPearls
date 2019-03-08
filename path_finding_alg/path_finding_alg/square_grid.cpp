#include "square_grid.h"

std::array<GridLocation, 4> SquareGrid::m_dirs =
{
	GridLocation{ 1, 0 },
	GridLocation{ 0, -1 },
	GridLocation{ -1, 0 },
	GridLocation{ 0, 1 }
};

bool SquareGrid::inBounds(GridLocation id) const
{
	return 0 <= id.x && id.x < m_width
		&& 0 <= id.y && id.y < m_height;
}

bool SquareGrid::passable(GridLocation id) const
{
	return m_walls.find(id) == m_walls.end();
}

std::vector<GridLocation> SquareGrid::neighbors(GridLocation id) const
{
	std::vector<GridLocation> results;
	results.reserve(m_dirs.size());
	for (GridLocation dir : m_dirs)
	{
		GridLocation next{ id.x + dir.x, id.y + dir.y };
		if (inBounds(next) && passable(next))
		{
			results.push_back(next);
		}
	}

	if ((id.x + id.y) % 2 == 0)
	{
		std::reverse(results.begin(), results.end());
	}

	return results;
}

void SquareGrid::addBlocks(int xMin, int yMin, int xMax, int yMax)
{
	for (int x = xMin; x < xMax; ++x)
	{
		for (int y = yMin; y < yMax; ++y)
		{
			m_walls.insert(GridLocation{ x, y });
		}
	}
}
