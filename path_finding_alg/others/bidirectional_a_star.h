#pragma once

#include <vector>
#include <list>
using std::list;
using std::vector;

class Graph
{
	int m_nodeCount;
	vector<list<int>> m_adjacencyList;
	
	int m_start, m_end;

	// boolean array for BFS started from 
	// source and target(front and backward BFS) 
	// for keeping track on visited nodes 
	vector<bool> m_startVisitedNodes, m_endVisitedNodes;

	// Keep track on parents of nodes 
	// for front and backward search 
	vector<int> s_parent, t_parent;

public:
	Graph(int n);
	int isIntersecting();
	void addEdge(int u, int v);
	void printPath(int intersectNode);
	void BFS(list<int>& queue, vector<bool>& visited, vector<int>& parent);
	int biDirSearch(int src, int dest);
};
