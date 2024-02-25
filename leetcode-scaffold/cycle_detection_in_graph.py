#!/usr/bin/env python

from collections import defaultdict

class Graph(object):
    def __init__(self, nodeCount):
        self.G = defaultdict(list)
        self.V = nodeCount

    def addEdge(self, u, v):
        self.G[u].append(v)

    def isCyclic(self):
        #return self.isCyclic_dfs()
        return self.isCyclic_colors()

    def isCyclic_colors(self):
        def dfs(u):
            colors[u] = 1
            for v in self.G[u]:
                if colors[v] == 1:
                    print("cycle detected: ({}, {})".format(u, v))
                    print(colors)
                    return True
                elif colors[v] == 0 and dfs(v):
                    return True
            colors[u] = 2
            return False

        # 0 - unvisited, 1 - visiting, 2 - visited
        colors = [0] * self.V
        for i in range(self.V):
            if colors[i] != 0:
                continue
            if dfs(i):
                return True
        return False

    def isCyclic_dfs(self):
        def dfs(u):
            visited[u] = True
            onStack[u] = True
            for v in self.G[u]:
                if not visited[v] and dfs(v):
                    return True
                elif onStack[v]:
                    print("cycle detected: ({}, {})".format(u, v))
                    print(onStack)
                    return True
            onStack[u] = False
            return False

        visited = [False] * self.V
        onStack = [False] * self.V

        for i in range(self.V):
            if visited[i]:
                continue
            if dfs(i):
                return True
        return False


if __name__ == '__main__':
    graph = Graph(5)
    graph.addEdge(0, 1)
    graph.addEdge(0, 2)
    graph.addEdge(1, 2)
    graph.addEdge(2, 0)
    graph.addEdge(2, 3)
    graph.addEdge(3, 3)

    if graph.isCyclic():
        print("cycle detected")
    else:
        print("cycle not found")

