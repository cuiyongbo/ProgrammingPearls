#include "bidirectional_a_star.h"

#include <vector>
#include <iostream>
#include <iterator>
using namespace std;

Graph::Graph(int n)
{
	m_nodeCount = n;
	m_adjacencyList.resize(n);
};

// Method for adding undirected edge 
void Graph::addEdge(int u, int v)
{
	m_adjacencyList[u].push_back(v);
	m_adjacencyList[v].push_back(u);
};

// Method for Breadth First Search 
void Graph::BFS(list<int>& queue, vector<bool>& visited, vector<int>& parent)
{
	int current = queue.front();
	queue.pop_front();

	for (int n: m_adjacencyList[current])
	{
		if (!visited[n])
		{
			parent[n] = current;
			visited[n] = true;
			queue.push_back(n);
		}
	}
};

// check for intersecting vertex 
int Graph::isIntersecting()
{
	int intersectNode = -1;
	for (int i = 0; i < m_nodeCount; i++)
	{
		if (m_startVisitedNodes[i] && m_endVisitedNodes[i])
		{
			intersectNode = i;
			break;
		}
	}
	return intersectNode;
};

// Print the path from source to target 
void Graph::printPath(int intersectNode)
{
	vector<int> path;
	path.push_back(intersectNode);
	int i = intersectNode;
	while (i != m_start)
	{
		path.push_back(s_parent[i]);
		i = s_parent[i];
	}
	reverse(path.begin(), path.end());

	i = intersectNode;
	while (i != m_end)
	{
		path.push_back(t_parent[i]);
		i = t_parent[i];
	}

	cout << "*****Path*****\n";
	copy(path.begin(), path.end(), ostream_iterator<int>(cout, " "));
	cout << "\n";
};

int Graph::biDirSearch(int src, int dest)
{
	m_start = src;
	m_end = dest;

	m_startVisitedNodes.assign(m_nodeCount, false);
	m_endVisitedNodes.assign(m_nodeCount, false);

	s_parent.assign(m_nodeCount, -1);
	t_parent.assign(m_nodeCount, -1);

	// queue for front and backward search 
	list<int> s_queue, t_queue;

	s_queue.push_back(m_start);
	m_startVisitedNodes[m_start] = true;

	t_queue.push_back(m_end);
	m_endVisitedNodes[m_end] = true;

	// parent of parent, target is set to -1 
	t_parent[m_end] = -1;
	s_parent[m_start] = -1;

	while (!s_queue.empty() && !t_queue.empty())
	{
		// Do BFS from source and target vertices 
		BFS(s_queue, m_startVisitedNodes, s_parent);
		BFS(t_queue, m_endVisitedNodes, t_parent);

		// check for intersecting vertex 
		int intersectNode = isIntersecting();
		if (intersectNode != -1)
		{
			cout << "Path exist between " << m_start << " and " << m_end << "\n";
			cout << "Intersection at: " << intersectNode << "\n";
			printPath(intersectNode);
			return 0;
		}
	}
	return -1;
}

int a_star_test()
{
	int s = 0;
	int t = 14;

	int n = 15;
	Graph g(n);
	g.addEdge(0, 4);
	g.addEdge(1, 4);
	g.addEdge(2, 5);
	g.addEdge(3, 5);
	g.addEdge(4, 6);
	g.addEdge(5, 6);
	g.addEdge(6, 7);
	g.addEdge(7, 8);
	g.addEdge(8, 9);
	g.addEdge(8, 10);
	g.addEdge(9, 11);
	g.addEdge(9, 12);
	g.addEdge(10, 13);
	g.addEdge(10, 14);
	if (g.biDirSearch(s, t) == -1)
		cout << "Path don't exist between " << s << " and " << t << "\n";

	return 0;
}

