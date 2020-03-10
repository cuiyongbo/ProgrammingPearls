#pragma once

#include <iostream>
#include <tuple>
#include <vector>
#include <unordered_map>

// an adjacency list notation
struct SimpleGraph
{
	std::unordered_map<char, std::vector<char> > edges;
	std::vector<char>& neighbors(char id) { return edges[id]; }
};

struct GridLocation
{
	int x, y;
};

inline bool operator==(const GridLocation& l, const GridLocation& r)
{
	return l.x == r.x && l.y == r.y;
}

inline bool operator!=(const GridLocation& l, const GridLocation& r)
{
	return !(l == r);
}

inline bool operator<(const GridLocation& l, const GridLocation& r)
{
	// compares l.x to r.x then l.y ot r.y
	return std::tie(l.x, l.y) < std::tie(r.x, r.y);
}

inline std::ostream& operator<< (std::ostream& out, const GridLocation& loc)
{
	out << '(' << loc.x << ',' << loc.y << ')';
	return out;
}

namespace std
{
	template<> struct hash<GridLocation>
	{
		//typedef GridLocation argument_type;
		std::size_t operator()(const GridLocation& id) const
		{
			return std::hash<int>()(id.x ^ (id.y << 4));
		}
	};
}

